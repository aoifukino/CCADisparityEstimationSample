using OpenCvSharp;
using System.Threading.Tasks;

namespace LabDispModules;
public enum EdgeDirection
{
    Horizontal,
    Vertical,
    Both
}

public static partial class LabDispModule
{
    public static Mat DetectEdge(Mat src, MatType type, EdgeDirection direction = EdgeDirection.Both, double thValue = 0)
    {
        var dst = new Mat();
        var kernel_x = new double[3, 3]
        {
            { 0, -1, 0 },
            { 0 , 0 , 0},
            { 0, 1 , 0},
        };

        var kernel_y = new double[3, 3]
        {
            { 0, 0, 0 },
            { -1, 0, 1},
            { 0, 0, 0},
        };
        var Gx = new Mat();
        var Gy = new Mat();
        Cv2.Filter2D(src, Gx, MatType.CV_64FC1, InputArray.Create(kernel_x));
        Cv2.Filter2D(src, Gy, MatType.CV_64FC1, InputArray.Create(kernel_y));

        switch (direction)
        {
            case EdgeDirection.Both:
                Cv2.Pow(Gx, 2, Gx);
                Cv2.Pow(Gy, 2, Gy);
                Cv2.Sqrt(Gx + Gy, dst);
                break;
            case EdgeDirection.Horizontal:

                dst = Gx.Abs();
                break;
            case EdgeDirection.Vertical:
                dst = Gy.Abs();
                break;
            default:
                throw new ArgumentException("");
        }
        dst.ConvertTo(dst, type);
        Cv2.Threshold(dst, dst, thValue, 255, ThresholdTypes.Tozero);

        return dst;
    }

    public static Mat SobelEdgeDetection(Mat src,MatType type,int ksize = 3){
        var dst = new Mat();
        var Gx = new Mat();
        var Gy = new Mat();
        Cv2.Sobel(src, Gx, MatType.CV_64FC1, 1, 0,ksize:ksize);
        Cv2.Sobel(src, Gy, MatType.CV_64FC1, 0, 1,ksize:ksize);
        Cv2.Pow(Gx, 2, Gx);
        Cv2.Pow(Gy, 2, Gy);
        Cv2.Sqrt(Gx + Gy, dst);
        dst.ConvertTo(dst, type);
        return dst;
    }
}
