using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace LabDispModules;

public static partial class LabDispModule
{
    public static Mat ApplyColorMapWithColorBar(Mat src, ColormapTypes type, int minValue, int maxValue, int numStep = 10)
    {
        var num_bar_w = 60;
        var color_bar_w = 50;
        var vline = 30;

        var grayColorMap = new Mat();
        var colorMap = new Mat();

        grayColorMap = ((double)255 / (maxValue - minValue)) * src + (-(double)255 * minValue / (maxValue - minValue));
        grayColorMap.ConvertTo(grayColorMap, MatType.CV_8UC1);
        colorMap = ExchangeColor(grayColorMap);

        var colorBar = new Mat(new Size(color_bar_w, grayColorMap.Height), MatType.CV_8UC1, new Scalar(255));

        Parallel.For(0, colorBar.Height, y => {
            for (int x = 0; x < colorBar.Width; x++)
            {
                int value = 255 - (255 * y) / colorBar.Height;
                colorBar.At<byte>(y, x) = (byte)value;

            }
        });

        var numWindow = new Mat(new Size(num_bar_w, grayColorMap.Height), MatType.CV_8UC3, new Scalar(255, 255, 255));
        for (int i = 0; i <= maxValue; i += numStep)
        {
            int y = i * numWindow.Height / maxValue;
            Cv2.PutText(numWindow, i.ToString(), new Point(5, numWindow.Height - y), HersheyFonts.HersheySimplex, 0.5, new Scalar(0, 0, 0), 1, LineTypes.Link4, false);
        }
        colorBar = ExchangeColor(colorBar);

        var winMat = new Mat(new Size(grayColorMap.Width + num_bar_w + num_bar_w + vline, grayColorMap.Height), MatType.CV_8UC3, new Scalar(255, 255, 255));
        colorMap.CopyTo(new Mat(winMat, new Rect(0, 0, grayColorMap.Width, grayColorMap.Height)));
        numWindow.CopyTo(new Mat(winMat, new Rect(grayColorMap.Width + vline + color_bar_w, 0, num_bar_w, grayColorMap.Height)));
        colorBar.CopyTo(new Mat(winMat, new Rect(grayColorMap.Width + vline, 0, color_bar_w, grayColorMap.Height)));
        return winMat;
    }
}
