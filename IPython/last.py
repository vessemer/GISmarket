test_df = pd.read_csv(os.path.join(config.PATHS['CSV'], 'sample_submission.csv'))
for name in config.label_names_list + ['Predicted', 'Target']:
    test_df[name] = 0


def prepare_learner(fold, checkpoint):
    model = models.resnet101(pretrained=True)
    model = lrn.set_io_dims(model, in_channels=4, out_channels=28, dropout=PARAMS['DROPOUT'])
    model = lrn.get_model(model, checkpoint=checkpoint, devices=PARAMS['CUDA_DEVICES'])
    model.module.eval()

    loss = FocalLoss(gamma=2)

    opt = torch.optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=PARAMS['EXP_GAMMA'], last_epoch=-1)
    learner = lrn.Learner(model=model, opt=opt, loss=loss)
    return learner

def prepare_data(fold):
    folds = ds.get_folds(config.PARAMS['NB_FOLDS'])
    _, valid_dataset = ds.get_datasets(folds, fold)

    test_dataset = ds.ProteinDataset(
        test_df, 
        config.PATHS['TEST'], 
        config.label_names_list, 
        augmentations=None
    )
    return valid_dataset, test_dataset


def infer(model, idx, data, predicts, labels=None):
    with torch.no_grad():
        image = data['image'].unsqueeze(0)
        image = augs.get_crops(image, config.PARAMS['SIDE'])
        image = augs._rotate_mirror_do(image)

        image = torch.tensor(image)
        image = torch.autograd.Variable(image).cuda()
        predict = model(image)
        predict = torch.sigmoid(predict)
        predict = predict.reshape(8, 5, len(config.label_names_list))
        predicts[idx] = predict.mean(0).data.cpu().numpy()

        image = image.data.cpu()
        if labels is not None:
            labels[idx] = data['label']


def orchestrate_inference(fold, checkpoints_pth, epoch):
    template_path = 'fold_{}_checkpoint.epoch_{}'
    path = os.path.join(checkpoints_pth, template_path)
    dump_path = os.path.join(checkpoints_pth, '{}_fold_{}')
    
    history = pickle.load(open(path.format(fold, 'loss'), 'rb'))
    plt.figure(figsize=(15,10))
    sns.barplot(y=label_names_list, x=history['valid_losses'][epoch]['f1_score']);
    plt.show()

    formated_train = format_history(history['train_losses'])
    formated_valid = format_history(history['valid_losses'])
    plot_losses(formated_train, formated_valid)
    
    learner = prepare_learner(fold, path.format(fold, epoch))
    valid_dataset, test_dataset = prepare_data(fold)
    

    predicts = np.zeros(shape=(len(test_dataset), 5, len(config.label_names_list)))
    try:
        for idx, data in tqdm(enumerate(test_dataset)):
            infer(learner.model, idx, data, predicts, labels=None)
    except Exception as e:
        print(e)
        print('idx: {}'.format(idx))
    np.save(dump_path.format('test_predicts', fold), predicts)
    test_dataset.keys.to_csv(dump_path.format('test_keys', fold))


    valid_predicts = np.zeros(shape=(len(valid_dataset), 5, len(config.label_names_list)))
    valid_labels = np.zeros(shape=(len(valid_dataset), len(config.label_names_list)))
    try:
        for idx, data in tqdm(enumerate(valid_dataset)):
            infer(learner.model, idx, data, valid_predicts, labels=valid_labels)
    except Exception as e:
        print(e)
        print('idx: {}'.format(idx))
    np.save(dump_path.format('valid_predicts', fold), valid_predicts)
    np.save(dump_path.format('valid_labels', fold), valid_labels)
    valid_dataset.keys.to_csv(dump_path.format('valid_keys', fold))


epoch = 49
checkpoints_pth = '../data/models/resnet101_side_384_pilo/'


for fold in range(config.PARAMS['NB_FOLDS']):
    orchestrate_inference(fold, checkpoints_pth, epoch)


dump_path = os.path.join(checkpoints_pth, '{}_fold_{}')
template_path = 'fold_{}_checkpoint.epoch_{}'
path = os.path.join(checkpoints_pth, template_path)


valid_predicts = list()
valid_labels = list()
for fold in range(config.PARAMS['NB_FOLDS']):
    valid_predicts.append(np.load(dump_path.format('valid_predicts', fold) + '.npy'))
    valid_labels.append(np.load(dump_path.format('valid_labels', fold) + '.npy'))
valid_predicts = np.concatenate(valid_predicts)
valid_labels = np.concatenate(valid_labels)


predicts = list()
for fold in range(config.PARAMS['NB_FOLDS']):
    predicts.append(np.load(dump_path.format('test_predicts', fold) + '.npy'))
predicts = np.stack(predicts)

