from mmcv.runner.hooks import HOOKS, OptimizerHook


@HOOKS.register_module()
class CustomOptimizerHook(OptimizerHook):
    """Optimizer hook that fails fast on invalid backward passes."""

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)

        loss = runner.outputs['loss']
        if not loss.requires_grad:
            raise RuntimeError(
                'The total loss does not require gradients. Check whether all '
                'loss terms are detached or replaced by constant tensors.')
        loss.backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                        runner.outputs['num_samples'])
        runner.optimizer.step()
