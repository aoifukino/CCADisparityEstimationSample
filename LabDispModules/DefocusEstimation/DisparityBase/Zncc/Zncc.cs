using OpenCvSharp;
using System.Numerics;



namespace LabDispModules;
public static partial class LabDispModule
{
    /// <summary>
    /// ZNCCによる色ずれ量（視差）計算
    /// </summary>
    /// <param name="stdImg"></param>
    /// <param name="tmpImg">テンプレート画像</param>
    /// <param name="kernelSize">カーネルサイズ</param>
    /// <param name="searchRange_x"></param>
    /// <returns></returns>
    public static Mat CalcDispByZNCC(Mat stdImg, Mat tmpImg, int kernelSize, int searchRange_x)
    {
        if(stdImg.Channels() != 1 || tmpImg.Channels() != 1) throw new Exception("入力画像は1チャンネルの画像を入力してください");
        int kSize = (kernelSize - 1) / 2;
        int sRange = (searchRange_x - 1) / 2;
        var dMap = Mat.Zeros(stdImg.Size(), MatType.CV_32FC1).ToMat();
        var edge = DetectEdge(stdImg, MatType.CV_8UC1, EdgeDirection.Both, 15);
        var canny = new Mat();
        Cv2.Canny(stdImg, canny, 50, 100);

        var edgeStdImg = new Mat();
        var edgeTmpImg = new Mat();
        Cv2.Sobel(stdImg, edgeStdImg, MatType.CV_32FC1, 1, 0);
        Cv2.Sobel(tmpImg, edgeTmpImg, MatType.CV_32FC1, 1, 0);

        edgeStdImg = edgeStdImg.Abs();
        edgeTmpImg = edgeTmpImg.Abs();

        edgeStdImg.ConvertTo(edgeStdImg, MatType.CV_8UC1);
        edgeTmpImg.ConvertTo(edgeTmpImg, MatType.CV_8UC1);

        Parallel.For(kSize, stdImg.Height - kSize, y =>
        {
            for (int x = sRange + kSize + 1; x < stdImg.Width - sRange - kSize - 1; x++)
            {
                if (edge.At<byte>(y, x) > 0)
                {
                    //中央
                    var roi_r = new Mat(edgeStdImg, new Rect(x - kSize, y - kSize, kernelSize, kernelSize));


                    var maxSim = double.MinValue;
                    int max_x = 0;
                    var simArray = new float[searchRange_x];

                    for (int shift_x = 0; shift_x < searchRange_x; shift_x++)
                    {
                        var matchMat = new Mat();
                        var roi_b = new Mat(edgeTmpImg, new Rect(x + sRange - kSize - shift_x, y - kSize, kernelSize, kernelSize));
                        Cv2.MatchTemplate(roi_r, roi_b, matchMat, TemplateMatchModes.CCoeffNormed);

                        double maxVal;
                        Cv2.MinMaxLoc(matchMat, out _, out maxVal);

                        simArray[shift_x] = (float)maxVal;
                        if (maxVal > maxSim && shift_x != 0 && shift_x != searchRange_x - 1)
                        {
                            maxSim = maxVal;
                            max_x = shift_x;
                        }
                    }

                    var Rneg1 = simArray[max_x - 1];
                    var R = simArray[max_x];
                    var Rpos1 = simArray[max_x + 1];
                    float subpixel = 0;


                    var denominator = 4 * R - 2 * Rneg1 - 2 * Rpos1;
                    if (denominator != 0)
                    {
                        subpixel = (Rpos1 - Rneg1) / denominator;
                    }


                    var submin = max_x + subpixel;
                    // var absSubmin = subpixel == 0 ? 10 : 0;
                    var absSubmin = Math.Abs(submin - sRange);
                    dMap.Set<float>(y, x, absSubmin);

                }
            }
        });
        return dMap;
    }
}
