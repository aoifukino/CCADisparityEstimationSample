using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace LabDispModules;

public static partial class LabDispModule
{
    public static Mat ExchangeColor(Mat src)
    {
        if (src.Type() != MatType.CV_8UC1) throw new System.Exception("MatType is not CV_8UC1");

        var dst = Mat.Zeros(src.Height, src.Width, MatType.CV_8UC3).ToMat();

        Parallel.For(0, src.Height, y =>
        {
            for (int x = 0; x < dst.Width; x++)
            {
                var value = src.At<byte>(y, x);
                var value_norm = (double)value / 255;
                //カラー変換
                if (value_norm >= 0 && value_norm <= 0.25)
                {
                    dst.Set<Vec3b>(y, x, new Vec3b(255, (byte)(255 * Math.Sin(value_norm * 2 * Math.PI)), 0));
                }
                else if (value_norm > 0.25 && value_norm <= 0.5)
                {
                    dst.Set<Vec3b>(y, x, new Vec3b((byte)(255 * Math.Sin(value_norm * 2 * Math.PI)), 255, 0));
                }
                else if (value_norm > 0.5 && value_norm <= 0.75)
                {
                    dst.Set<Vec3b>(y, x, new Vec3b(0, 255, (byte)(-255 * Math.Sin(value_norm * 2 * Math.PI))));
                }
                else
                {
                    dst.Set<Vec3b>(y, x, new Vec3b(0, (byte)(-255 * Math.Sin(value_norm * 2 * Math.PI)), 255));
                }
            }
        });
        return dst;
    }
}
