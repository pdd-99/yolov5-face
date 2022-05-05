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
                # Brightness
                iaa.Add(iap.Normal(iap.Choice([-10, 10]), 5)),
                iaa.AddToBrightness((-15, 15)),
                iaa.MultiplyBrightness((0.95, 1.05)),
                iaa.MultiplyAndAddToBrightness(mul=(0.85, 1.05), add=(-5, 5)),
                
                # Change contrast
                iaa.SigmoidContrast(gain= (4, 6), cutoff=(0.5, 0.6)),
                iaa.LinearContrast((0.5, 1.15)),
                iaa.GammaContrast((0.5, 2)),
                iaa.LogContrast(gain=(0.75, 1.0)),
                iaa.pillike.Autocontrast((1, 3))
            ]),
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

        # Temperature 
        iaa.Sometimes(0.1, iaa.Multiply((0.9, 1.1))),
    ]
)
