using OpenCvSharp;
using System.Numerics;

namespace LabDispModules;

public static partial class LabDispModule
{

    public static Memory<UInt128> CensusTransform2(this Mat src, int kSize)
    {
        var census = new UInt128[src.Height*src.Width];
        var censusMemory = census.AsMemory();
        int width = src.Width;
        UInt128 code = 0;
        Parallel.For(kSize, src.Height - kSize, y =>
        {
            var span = censusMemory.Span.Slice(y * width, width);
            for (int x = kSize; x < src.Width - kSize; x++)
            {
                for (int sy = 0; sy < src.Height; sy++)
                {
                    for (int sx = 0; sx < src.Width; sx++)
                    {
                        byte center = src.At<byte>(kSize, kSize);
                        if (y == kSize && x == kSize)
                        {
                            continue;
                        }

                        byte neighbor = src.At<byte>(y, x);
                        code <<= 1;
                        code |= (byte)(neighbor >= center ? 1 : 0);
                    }
                }
                span[y * width + x] = code;
            }
        });
        return censusMemory;
    }

    public static UInt128 CensusTransform(this Mat template)
    {
        var kSize = (template.Width - 1) / 2;
        UInt128 code = 0;
        for (int y = 0; y < template.Height; y++)
        {
            for (int x = 0; x < template.Width; x++)
            {
                byte center = template.At<byte>(kSize, kSize);
                if (y == kSize && x == kSize)
                {
                    continue;
                }

                byte neighbor = template.At<byte>(y, x);
                code <<= 1;
                code |= (byte)(neighbor >= center ? 1 : 0);
            }
        }
        return code;
    }

    /// <summary>
    /// Censusによる色ずれ量（視差）計算
    /// </summary>
    /// <param name="stdImg"></param>
    /// <param name="tmpImg"></param>
    /// <param name="kernelSize"></param>
    /// <param name="searchRange_x"></param>
    /// <returns></returns>
    public static Mat CalcDispByCensus(Mat stdImg, Mat tmpImg, int kernelSize, int searchRange_x)
    {

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


        UInt128 maxUInt64 = UInt64.MaxValue;

        Parallel.For(kSize, stdImg.Height - kSize, y =>
        {
            for (int x = sRange + kSize + 1; x < stdImg.Width - sRange - kSize - 1; x++)
            {
                if (edge.At<byte>(y, x) > 0)
                {
                    //中央
                    var roi_r = new Mat(edgeStdImg, new Rect(x - kSize, y - kSize, kernelSize, kernelSize));

                    var census_r = roi_r.CensusTransform();

                    int minHamming = int.MaxValue;
                    int min_x = 0;
                    var lossArray = new int[searchRange_x];
                    for (int shift_x = 0; shift_x < searchRange_x; shift_x++)
                    {
                        var roi_b = new Mat(edgeTmpImg, new Rect(x + sRange - kSize - shift_x, y - kSize, kernelSize, kernelSize));
                        var census_b = roi_b.CensusTransform();

                        var xor_b_r = census_b ^ census_r;


                        var xorHead_b_r = (ulong)(xor_b_r >> 64);
                        var xorBack_b_r = (ulong)(xor_b_r & maxUInt64);

                        var hamming = BitOperations.PopCount(xorHead_b_r) + BitOperations.PopCount(xorBack_b_r);
                        lossArray[shift_x] = hamming;
                        if (hamming < minHamming && shift_x != 0 && shift_x != searchRange_x - 1)
                        {
                            minHamming = hamming;
                            min_x = shift_x;
                        }
                    }

                    var Rneg1 = lossArray[min_x - 1];
                    var R = lossArray[min_x];
                    var Rpos1 = lossArray[min_x + 1];
                    float subpixel = 0;

                    float denominator = 4 * R - 2 * Rneg1 - 2 * Rpos1;
                    if (denominator != 0)
                    {
                        subpixel = (Rpos1 - Rneg1) / denominator;
                    }

                    var submin = min_x + subpixel;
                    var absSubmin = Math.Abs(submin - sRange);
                    dMap.Set<float>(y, x, absSubmin);
                }
            }
        });
        return dMap;
    }
}
