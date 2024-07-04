import segmentation_models_pytorch as smp

model = smp.MAnet(
    encoder_name='mit_b5',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
)