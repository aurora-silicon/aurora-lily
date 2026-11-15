use super::*;
use crate::kernels::elementwise::copy_words;
use crate::tensor::DType;

fn words(ctx: &MetalContext, n: usize, value: u32) -> Tensor {
    let t = Tensor::zeros(ctx, &[n], DType::U32).expect("tensor");
    t.write_bytes(bytemuck::cast_slice(&vec![value; n])).expect("write");
    t
}

/// A pass put aside runs nothing until it is attached and committed, and
/// only on the context it was encoded on.
#[test]
fn detached_pass_runs_when_attached_and_committed() {
    let ctx = MetalContext::new().expect("context");
    let (src, dst) = (words(&ctx, 64, 7), words(&ctx, 64, 0));
    let pass = ctx.begin_concurrent().expect("pass");
    copy_words(&ctx, &pass, &src, &dst).expect("copy");
    let detached = pass.end().expect("end").detach();
    let begun = ctx.passes_begun();

    // Another pass runs and completes in between; the one put aside did not.
    let other = words(&ctx, 64, 0);
    let between = ctx.begin().expect("pass");
    copy_words(&ctx, &between, &src, &other).expect("copy");
    between.commit_wait().expect("wait");
    assert_eq!(other.to_u32().expect("read"), vec![7; 64]);
    assert_eq!(dst.to_u32().expect("read"), vec![0; 64]);
    assert_eq!(ctx.passes_begun(), begun + 1);

    detached.attach(&ctx).expect("attach").commit().expect("commit").wait().expect("wait");
    assert_eq!(dst.to_u32().expect("read"), vec![7; 64]);

    let elsewhere = MetalContext::new().expect("second context");
    let pass = ctx.begin().expect("pass");
    copy_words(&ctx, &pass, &src, &dst).expect("copy");
    let detached = pass.end().expect("end").detach();
    assert!(detached.attach(&elsewhere).is_err());
}

/// A short pass is one command buffer per part between its waits and
/// signals; any other goes in batches.
#[test]
fn short_pass_is_one_command_buffer_per_part() {
    let ctx = MetalContext::new().expect("context");
    let (src, dst) = (words(&ctx, 16, 3), words(&ctx, 16, 0));
    let event = ctx.new_shared_event().expect("event");
    let dispatches = 2 * CONCURRENT_BATCH + 44;
    let encode = |short: bool| {
        let pass = ctx.begin_concurrent().expect("pass");
        if short {
            pass.set_short();
        }
        for _ in 0..dispatches {
            copy_words(&ctx, &pass, &src, &dst).expect("copy");
            pass.memory_barrier().expect("barrier");
        }
        pass.signal_done(&event, if short { 2 } else { 1 }).expect("done");
        pass.end().expect("end")
    };
    let cmds = |pass: &EncodedPass<'_>| {
        pass.program.iter().filter(|item| matches!(item, Item::Cmd(..))).count()
    };
    let batched = encode(false);
    assert_eq!(cmds(&batched), 3);
    batched.commit().expect("commit").wait_polled().expect("wait");
    assert_eq!(event.signaled_value(), 1);
    let short = encode(true);
    assert_eq!(cmds(&short), 1);
    short.commit().expect("commit").wait_polled().expect("wait");
    assert_eq!(event.signaled_value(), 2);
    assert_eq!(dst.to_u32().expect("read"), vec![3; 16]);
}

/// Command pools given back are reset and reused, whether their pass ran,
/// was dropped encoded, or was put aside and dropped.
#[test]
fn command_pools_are_reused() {
    let ctx = MetalContext::new().expect("context");
    let (src, dst) = (words(&ctx, 16, 5), words(&ctx, 16, 0));
    for round in 0..24 {
        let pass = ctx.begin().expect("pass");
        copy_words(&ctx, &pass, &src, &dst).expect("copy");
        match round % 3 {
            0 => pass.commit_wait().expect("wait"),
            1 => drop(pass.end().expect("end")),
            _ => drop(pass.end().expect("end").detach()),
        }
    }
    let pools = ctx.pool.free.borrow().len() + ctx.pool.used.borrow().len();
    assert!(pools <= 2, "{pools} command pools for one pass at a time");
    assert_eq!(dst.to_u32().expect("read"), vec![5; 16]);
}

/// The capability scan reads the `OpCapability` run after the header and
/// stops at the first other instruction or a truncated one.
#[test]
fn spirv_capabilities_reads_the_leading_capabilities() {
    let header = [0x0723_0203, 0x0001_0600, 0, 100, 0];
    let cap = |c: u32| [(2 << 16) | 17, c];
    // OpExtension "ab" (10, 2 words), then a capability past it.
    let module: Vec<u32> = header
        .iter()
        .copied()
        .chain(cap(1))
        .chain(cap(5116))
        .chain([(2 << 16) | 10, 0x6261])
        .chain(cap(6022))
        .collect();
    assert_eq!(spirv_capabilities(&module).collect::<Vec<_>>(), vec![1, 5116]);
    assert_eq!(spirv_capabilities(&module[..8]).collect::<Vec<_>>(), vec![1]);
    assert_eq!(spirv_capabilities(&header).count(), 0);
}

/// Without matrix units a bfloat or cooperative-matrix kernel is refused
/// with an error (Mesa would fail compiling it or crash), and every other
/// kernel still builds.
#[test]
fn matrix_unit_kernels_are_refused_without_matrix_units() {
    let ctx = MetalContext::new().expect("context");
    for &(name, bytes) in KERNELS {
        let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).expect("spv");
        let needs = spirv_capabilities(&words).any(|c| NEEDS_MATRIX_UNITS.contains(&c));
        let module = ctx.dev.module(name);
        if needs && !ctx.matrix_units() {
            let err = module.expect_err(name).to_string();
            assert!(err.contains("needs matrix units"), "{name}: {err}");
        } else {
            module.unwrap_or_else(|e| panic!("{name}: {e:#}"));
        }
    }
}
