from mmcv.runner.hooks import Hook


class RandomFPHook(Hook):
    '''
    Shuffle false patchs in training
    '''

    def after_train_epoch(self, runner):
        dataset = runner.data_loader.dataset
        if not hasattr(dataset, 'add_random_fp'):
            return

        data_infos = dataset.add_random_fp()
        ori_infos = runner.data_loader.dataset.data_infos
        assert len(data_infos) == len(ori_infos)

        runner.data_loader.dataset.data_infos = data_infos
