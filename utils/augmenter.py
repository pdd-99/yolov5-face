import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap

augmenter = iaa.Sequential(
    [
        # Flip 
        iaa.Sometimes(0.4,
            iaa.OneOf([
                iaa.Fliplr(0.5), 
                iaa.Flipud(0.5)])
        ),

        iaa.Sometimes(0.3, 
            iaa.OneOf([
                iaa.Rotate(rotate=90, order=[0, 1], cval=(0,255), mode='constant'),
                iaa.Rotate(rotate=-90, order=[0, 1], cval=(0,255), mode='constant'),
            ])
        ),

        # iaa.Sometimes(0.3,
        #     iaa.Resize({"shorter-side": (480, 1080), "longer-side": "keep-aspect-ratio"}, interpolation=["linear", "cubic", "nearest", "area"])
        # ),

        # Noise
        iaa.Sometimes(0.2, 
            iaa.OneOf([
                iaa.AddElementwise((-10, 10)),
                iaa.AdditiveLaplaceNoise(scale=(0.005*255, 0.02*255), per_channel=0.5),
                iaa.AdditivePoissonNoise((1, 5)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5), 
            ])
        ),

        # Brightness + Color + Contrast
        iaa.Sometimes(0.2, 
            iaa.OneOf([
                iaa.Add(iap.Normal(iap.Choice([-25, 25]), 13)),
                iaa.Multiply((0.75, 1.25)),
                iaa.AddToBrightness((-30, 30)),
                iaa.MultiplyBrightness((0.75, 1.25)),
                iaa.MultiplyAndAddToBrightness(mul=(0.75, 1.25), add=(-20, 20)),
                iaa.BlendAlphaHorizontalLinearGradient(iaa.Add(iap.Normal(iap.Choice([-30, 30]), 20)), start_at=(0, 0.25), end_at=(0.75, 1)),
                iaa.BlendAlphaHorizontalLinearGradient(iaa.Add(iap.Normal(iap.Choice([-30, 30]), 20)), start_at=(0.75, 1), end_at=(0, 0.25)),

                # Change contrast
                iaa.SigmoidContrast(gain=(3, 7), cutoff=(0.3, 0.6)),
                iaa.LinearContrast((0.75, 1.25)),
                iaa.GammaContrast((0.7, 1.5)),
                iaa.LogContrast(gain=(0.6, 1.4)),
                iaa.pillike.Autocontrast((2, 5)),
                iaa.Emboss(alpha=0.5, strength=1),
            ])    
        ),

        # Low resolution, compressed image
        iaa.Sometimes(0.15, 
            iaa.OneOf([
                iaa.imgcorruptlike.Pixelate(severity=(2, 4)),
                iaa.imgcorruptlike.JpegCompression(severity=(1, 2)),
                iaa.KMeansColorQuantization(n_colors=(230, 256)),
                iaa.UniformColorQuantization(n_colors=(30, 256)),
            ])
        ),

        # Low light condition
        iaa.Sometimes(0.18, 
            iaa.Sequential([
                iaa.JpegCompression(compression=(35, 90)),
                iaa.OneOf([
                    iaa.AdditivePoissonNoise((1, 10), per_channel=True),
                    iaa.AdditivePoissonNoise((1, 5)),
                    iaa.AdditiveLaplaceNoise(scale=(0.005*255, 0.02*255)),
                    iaa.AdditiveLaplaceNoise(scale=(0.005*255, 0.02*255), per_channel=True)
                ])
            ])
        ),

        # Normal blur
        iaa.Sometimes(0.1,
            iaa.OneOf([
                iaa.GaussianBlur((1, 2)),
                iaa.AverageBlur(k=(2, 5)),
                iaa.pillike.EnhanceSharpness(),
                iaa.pillike.FilterSmoothMore((10, 200)),
                iaa.MedianBlur(k=3),
                iaa.MotionBlur([3, 7], angle=(-70, 70)),
                iaa.imgcorruptlike.DefocusBlur(severity=(1, 2))
            ]),
        ),
        
        # Color
        iaa.Sometimes(0.2, 
            iaa.OneOf([
                iaa.ChangeColorTemperature((5000, 12000)),
                iaa.AddToHue((-10, 10)), 
                iaa.AddToHueAndSaturation((-20, 2), per_channel=True),
                iaa.AddToSaturation((-20, 2)),
                iaa.MultiplySaturation((0.5, 1.5)), 
            ]),
        ),

        # Channel shuffle
        iaa.ChannelShuffle(0.2),
        iaa.Sometimes(0.05, iaa.CoarseDropout(0.1, size_percent=0.0025, per_channel=1)),
        # Temperature 
        iaa.Sometimes(0.1, iaa.Multiply((0.9, 1.1))),
    ]
)
