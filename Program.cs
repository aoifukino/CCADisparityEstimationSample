using LabDispModules;
using OpenCvSharp;

var img = Cv2.ImRead(@"画像のパス");

// resize
Cv2.Resize(img, img, new Size(352, 528));
Console.WriteLine(img.Size());
var spimg = img.Split();
var red = spimg[0];
var green = spimg[1];
var blue = spimg[2];

Cv2.ImShow("img", img);
Cv2.WaitKey(0);

// var dMap = LabModule.CalcCCADefocus(src, 17);
// var censusDMap = LabModule.CalcDispByCensus(red, blue, 11, 21);
var znccDMap = LabDispModule.CalcDispByZNCC(red, blue, 17, 21);


var colorZnccDMap = LabDispModule.ApplyColorMapWithColorBar(znccDMap, ColormapTypes.Jet, 0, 5, 1);

Cv2.ImShow("colorZnccDMap", colorZnccDMap);
Cv2.WaitKey(0);