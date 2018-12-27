def create_model(opt,config):
    model = None
    print('Loading model %s...' % opt.trainer)

    if opt.trainer == 'UNIT':
        from .UNIT_trainer import UNIT_Trainer
        model = UNIT_Trainer(config)
    elif opt.trainer == 'MUNIT':
        from .MUNIT_trainer import MUNIT_Trainer
        model = MUNIT_Trainer(config)
    elif opt.trainer == 'myMUNIT':
        from .myMUNIT_trainer import myMUNIT_Trainer
        model = myMUNIT_Trainer(config)
    elif opt.trainer == 'myMUNIT_patch':
        from .myMUNIT_patch_trainer import myMUNIT_patch_Trainer
        model = myMUNIT_patch_Trainer(config)
    elif opt.trainer == 'myMUNIT_within_patch':
        from .myMUNIT_within_patch_trainer import myMUNIT_within_patch_Trainer
        model = myMUNIT_within_patch_Trainer(config)
    elif opt.trainer == 'myVAE_MUNIT_patch':
        from .myVAE_MUNIT_patch_trainer import my_VAE_MUNIT_patch_Trainer
        model = my_VAE_MUNIT_patch_Trainer(config)
    elif opt.trainer == 'myNet':
        from .myNet_trainer import myNet_Trainer
        model = myNet_Trainer(config)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    print("model [%s] was created" % (model.name()))
    return model