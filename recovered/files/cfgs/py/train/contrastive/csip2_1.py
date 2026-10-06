                    key_map_out=('image -> content_image',),
                ),
                image=ImageHandler(transform=T.Compose([
                        T.Resize((384, 384)),
                        T.RandomHorizontalFlip(),
                        WeakRandAugment2(),
                        T.ToTensor(),
                        T.Normalize(mean=ANIME_MEAN, std=ANIME_STD),
                    ]),
                ),
                
            ),
            bucket=CategoryBucket(target_size=384, min_cat_size=4),
        ),
        dataset_dechar=partial(BaseDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=StyleCharImageLabelSource(
                    img_root="/nfs/danbooru-w640-ws-full-artist/train",
                    char_label_file="/nfs/danbooru-w640-ws-full-artist/train_char_labels.json",
                    style_label_file="/nfs/danbooru-w640-ws-full-artist/train_split.json",
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                tagger=ImageHandler(
                    transform=load_preprocessor('animetimm/caformer_b36.dbv4-full'),
                    key_map_out=('image -> content_image',),
                ),
                image=ImageHandler(transform=T.Compose([
                        T.Resize((384, 384)),
                        T.RandomHorizontalFlip(),
                        WeakRandAugment2(),
                        T.ToTensor(),
                        T.Normalize(mean=ANIME_MEAN, std=ANIME_STD),
                    ]),
                ),
                key_map_out=('content_image -> content_image', 'image -> image', 'style -> label'),
            ),
            bucket=CharBucket(target_size=384, min_cat_size=8),
        )
    )

@neko_cfg
def cfg_evaluator():
    return partial(Evaluator,
        interval=2000,
        data_workers=8,
        metric=MetricGroup(
            AP=CXIPMetricContainer(AveragePrecision(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
            roc=CXIPMetricContainer(AUROC(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
            cohens=CXIPMetricContainer(cohens_d, key_map=('pred.pred -> pred', 'inputs.label -> label')),
        ),
        dataset=partial(BaseDataset, batch_size=64, loss_weight=1.0, 
            source=dict(
                data_source1=StyleImageLabelSource(
                    img_root="/nfs/danbooru-w640-ws-full-artist/train",
                    label_file="/nfs/danbooru-w640-ws-full-artist/test_split.json",
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                tagger=ImageHandler(
                    transform=load_preprocessor('animetimm/caformer_b36.dbv4-full'),
                    key_map_out=('image -> content_image',),
                ),
                image=ImageHandler(transform=T.Compose([
                        T.Resize((384, 384)),
                        T.ToTensor(),
                        T.Normalize(mean=ANIME_MEAN, std=ANIME_STD),
                    ]),
                )
            ),
            bucket=PosNegBucket(target_size=384, pos_rate=0.5),
        )
    )