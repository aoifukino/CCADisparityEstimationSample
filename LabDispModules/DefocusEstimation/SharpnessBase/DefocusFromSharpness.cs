using OpenCvSharp;

namespace LabDispModules;

public static partial class LabDispModule
{
    public static Mat DefocusFromSharpness(Mat src,double std1,double std2, double defocusThreshold = 1)
    {
        var blurdImg = new Mat();
        var reBlurdImg = new Mat();

        Cv2.GaussianBlur(src, blurdImg, new Size(9, 9), std1);
        Cv2.GaussianBlur(src, reBlurdImg, new Size(17, 17), std2);

        var bluredEdge = SobelEdgeDetection(blurdImg, MatType.CV_64FC1, ksize: 3); //ksize=3
        var reBlurdEdge = SobelEdgeDetection(reBlurdImg, MatType.CV_64FC1, ksize: 3);

        //canny
        // var canny = new Mat();
        // Cv2.Canny(green,canny,30,50);

        Cv2.Threshold(bluredEdge, bluredEdge, 2, 255, ThresholdTypes.Tozero);
        Cv2.Threshold(reBlurdEdge, reBlurdEdge, 2, 255, ThresholdTypes.Tozero);

        bluredEdge = bluredEdge.Abs();
        reBlurdEdge = reBlurdEdge.Abs();
        var ratio = new Mat();
        Cv2.Divide(bluredEdge, reBlurdEdge, ratio);

        var pow = new Mat();
        var sqrt = new Mat();
        Cv2.Pow(ratio, 2, pow);
        Cv2.Sqrt(pow - 1, sqrt);
        var sparseDMap = (std2 / sqrt).ToMat();


        Cv2.Threshold(sparseDMap, sparseDMap, defocusThreshold, 255, ThresholdTypes.Tozero);
        return sparseDMap;
    }
}