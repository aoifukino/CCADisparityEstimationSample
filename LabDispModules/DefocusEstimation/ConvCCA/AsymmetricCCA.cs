using OpenCvSharp;
using System.Runtime.CompilerServices;

namespace LabDispModules;

public static partial class LabDispModule
{
    /// <summary>
    /// 東芝の分割カラーフィルタを用いた距離推定
    /// https://www.global.toshiba/content/dam/toshiba/migration/corp/techReviewAssets/tech/review/2018/01/73_01pdf/f02.pdf
    /// </summary>
    /// <param name="src"></param>
    /// <param name="kernelSize"></param>
    /// <returns></returns>
    public static Mat CalcCCADefocus(Mat src, int kernelSize)
    {
        var bgr = src.Split();
        var blueImg = bgr[0];
        var greenImg = bgr[1];
        var redImg = bgr[2];

        var searchSigmaRange = 5f;

        var greenEdge = new Mat();
        var blueEdge = new Mat();
        var redEdge = new Mat();
        Cv2.Sobel(greenImg, greenEdge, MatType.CV_32FC1, 1, 0);
        Cv2.Sobel(blueImg, blueEdge, MatType.CV_32FC1, 1, 0);
        Cv2.Sobel(redImg, redEdge, MatType.CV_32FC1, 1, 0);
        greenEdge = greenEdge.Abs();
        blueEdge = blueEdge.Abs();
        redEdge = redEdge.Abs();
        greenEdge.ConvertTo(greenEdge, MatType.CV_8UC1);
        blueEdge.ConvertTo(blueEdge, MatType.CV_8UC1);
        redEdge.ConvertTo(redEdge, MatType.CV_8UC1);

        var result = Mat.Zeros(src.Size(), MatType.CV_32FC1).ToMat();
        var defocusMap = Mat.Zeros(src.Size(), MatType.CV_32FC1).ToMat();

        var sigmaMap = Mat.Zeros(src.Size(), MatType.CV_32FC1).ToMat();
        var k = (kernelSize - 1) / 2;
        var side = k + k - 1;

        int th = 30;
        Parallel.For(side, src.Height - side, y =>
       {
           for (int x = side; x < src.Width - side; x++)
           {
               if (greenEdge.At<byte>(y, x) < th || blueEdge.At<byte>(y, x) < th || redEdge.At<byte>(y, x) < th) continue;
               var rect = new Rect(x - side, y - side, kernelSize + 2 * k - 2, kernelSize + 2 * k - 2);
               var blueRect = new Mat(blueImg, rect);
               var greenRect = new Mat(greenImg, rect);
               var redRect = new Mat(redImg, rect);

               var low = 0f;
               var high = 100f;

               for (int cnt = 10; cnt >= 0; cnt--)
               {
                   float c1 = (low * 2 + high) / 3;
                   float c2 = (low + high * 2) / 3;

                   var c1Sim = Calculate(blueRect, redRect, greenRect, src, (float)(-searchSigmaRange + c1 * 0.1f), k, kernelSize);
                   var c2Sim = Calculate(blueRect, redRect, greenRect, src, (float)(-searchSigmaRange + c2 * 0.1f), k, kernelSize);

                   if (c1Sim > c2Sim) low = c1;
                   else high = c2;
               }

               var minSigma = (float)(-searchSigmaRange + low * 0.1f);
               defocusMap.At<float>(y, x) = Math.Abs(minSigma);
           }
       });
        return defocusMap;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Calculate(Mat blueRect, Mat redRect, Mat greenRect, Mat img, float sigma, int k, int kernelSize)
    {
        var blueKernelArray = GenerateAsymmetricKernel(kernelSize, sigma);
        var redKernelArray = GenerateReverseAsymmetricKernel(kernelSize, sigma);
        var normBlueKernel = blueKernelArray.Select(x => x / blueKernelArray.Sum()).ToArray();
        var normRedKernel = redKernelArray.Select(x => x / redKernelArray.Sum()).ToArray();
        var blueKernel = new Mat(1, normBlueKernel.Length, MatType.CV_64FC1, normBlueKernel);
        var redKernel = new Mat(1, normRedKernel.Length, MatType.CV_64FC1, normRedKernel);

        var bluredBlueImg = new Mat();
        var bluredRedImg = new Mat();
        //左右非対称PSFから左右対称なPSFに補正
        Cv2.Filter2D(blueRect, bluredBlueImg, MatType.CV_8UC1, blueKernel);
        Cv2.Filter2D(redRect, bluredRedImg, MatType.CV_8UC1, redKernel);

        var bluredBlueEdgeImg2 = new Mat();
        var bluredRedEdgeImg2 = new Mat();
        var greenEdgeImg2 = new Mat();
        Cv2.Sobel(bluredBlueImg, bluredBlueEdgeImg2, MatType.CV_32FC1, 1, 0);
        Cv2.Sobel(bluredRedImg, bluredRedEdgeImg2, MatType.CV_32FC1, 1, 0);
        Cv2.Sobel(greenRect, greenEdgeImg2, MatType.CV_32FC1, 1, 0);

        bluredBlueEdgeImg2 = bluredBlueEdgeImg2.Abs();
        bluredRedEdgeImg2 = bluredRedEdgeImg2.Abs();
        greenEdgeImg2 = greenEdgeImg2.Abs();

        bluredBlueEdgeImg2.ConvertTo(bluredBlueEdgeImg2, MatType.CV_8UC1);
        bluredRedEdgeImg2.ConvertTo(bluredRedEdgeImg2, MatType.CV_8UC1);
        greenEdgeImg2.ConvertTo(greenEdgeImg2, MatType.CV_8UC1);

        //カーネルのサイズに合わせて切り取り
        var blueRect2 = new Mat(bluredBlueEdgeImg2, new Rect(k - 1, k - 1, kernelSize, kernelSize));
        var redRect2 = new Mat(bluredRedEdgeImg2, new Rect(k - 1, k - 1, kernelSize, kernelSize));
        var greenRect2 = new Mat(greenEdgeImg2, new Rect(k - 1, k - 1, kernelSize, kernelSize));


        var b_r = new Mat();
        var b_g = new Mat();
        var r_g = new Mat();

        //マッチング
        Cv2.MatchTemplate(blueRect2, redRect2, b_r, TemplateMatchModes.CCoeffNormed);
        Cv2.MatchTemplate(blueRect2, greenRect2, b_g, TemplateMatchModes.CCoeffNormed);
        Cv2.MatchTemplate(redRect2, greenRect2, r_g, TemplateMatchModes.CCoeffNormed);

        double maxLocBR, maxLocBG, maxLocRG;
        Cv2.MinMaxLoc(b_r, out _, out maxLocBR);
        Cv2.MinMaxLoc(b_g, out _, out maxLocBG);
        Cv2.MinMaxLoc(r_g, out _, out maxLocRG);

        return (3 - maxLocBG - maxLocBR - maxLocRG);
    }

}