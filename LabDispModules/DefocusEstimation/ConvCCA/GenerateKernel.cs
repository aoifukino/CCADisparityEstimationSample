using OpenCvSharp;

namespace LabDispModules;
public record GenAsymmetrickernelParam{
    /// <summary>
    /// Rc/(A/2)
    /// </summary>
    public float Beta { get; init; }

    public float apertureDiameter { get; init; }

    public float focalLength { get; init; }

    public float Z_ipf { get; init; }

    public float PixelSize { get; init; }
}

public static partial class LabDispModule{
    public static double[] GenerateAsymmetricKernel(int width, float sigma, float scaleFac = 1.0f, float d = 5f)
    {
        if (width % 2 == 0) throw new Exception("widthは奇数で入力してください");
        var kernel = new Mat(1, width, MatType.CV_64FC1);
        var kArray = new double[width];
        int m = (width - 1) / 2;
        // float d = Math.Abs((genParam.Beta*genParam.focalLength*genParam.apertureDiameter*())/(genParam.Z_ipf*genParam.PixelSize));
        if (sigma > 0)
        {
            for (int x = 0; x < kernel.Width; x++)
            {
                var value = 1 / (sigma * Math.Sqrt(2 * Math.PI)) * Math.Exp(-0.5 * (((m - x) * (m - x)) / (sigma * sigma))) - 1 / (sigma * Math.Sqrt(2 * Math.PI)) * Math.Exp(-0.5 * (((m - x - d) * (m - x - d)) / (sigma * sigma)));
                kArray[x] = value > 0 ? value : 0;
            }
            return kArray;
        }
        else if (sigma < 0)
        {
            sigma *= -1;
            for (int x = 0; x < kernel.Width; x++)
            {
                var value = 1 / (sigma * Math.Sqrt(2 * Math.PI)) * Math.Exp(-0.5 * (((m - x) * (m - x)) / (sigma * sigma))) - 1 / (sigma * Math.Sqrt(2 * Math.PI)) * Math.Exp(-0.5 * (((m - x + d) * (m - x + d)) / (sigma * sigma)));
                kArray[x] = value > 0 ? value : 0;
            }
            return kArray;
        }
        else
        {
            sigma = 0.01f;
            for (int x = 0; x < kernel.Width; x++)
            {
                var value = 1 / (sigma * Math.Sqrt(2 * Math.PI)) * Math.Exp(-0.5 * (((m - x) * (m - x)) / (sigma * sigma)));
                kArray[x] = value > 0 ? value : 0;
            }
            return kArray;
        }
    }

    public static double[] GenerateReverseAsymmetricKernel(int width, float sigma,float d = 7f)
    {
        if (width % 2 == 0) throw new Exception("widthは奇数で入力してください");
        var kernel = new Mat(1, width, MatType.CV_64FC1);
        var kArray = new double[width];
        int m = (width - 1) / 2;

        if (sigma < 0)
        {
            sigma *= -1;
            for (int x = 0; x < kernel.Width; x++)
            {
                var value = 1 / (sigma * Math.Sqrt(2 * Math.PI)) * Math.Exp(-0.5 * (((m - x) * (m - x)) / (sigma * sigma))) - 1 / (sigma * Math.Sqrt(2 * Math.PI)) * Math.Exp(-0.5 * (((m - x - d) * (m - x - d)) / (sigma * sigma)));
                kArray[x] = value > 0 ? value : 0;
            }
            return kArray;
        }
        else if (sigma > 0)
        {
            for (int x = 0; x < kernel.Width; x++)
            {
                var value = 1 / (sigma * Math.Sqrt(2 * Math.PI)) * Math.Exp(-0.5 * (((m - x) * (m - x)) / (sigma * sigma))) - 1 / (sigma * Math.Sqrt(2 * Math.PI)) * Math.Exp(-0.5 * (((m - x + d) * (m - x + d)) / (sigma * sigma)));
                kArray[x] = value > 0 ? value : 0;
            }
            return kArray;
        }
        else
        {
            sigma = 0.01f;
            for (int x = 0; x < kernel.Width; x++)
            {
                var value = 1 / (sigma * Math.Sqrt(2 * Math.PI)) * Math.Exp(-0.5 * (((m - x) * (m - x)) / (sigma * sigma)));
                kArray[x] = value > 0 ? value : 0;
            }
            return kArray;
        }
    }
}