
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef struct intrinsic_param{
    int width;
    int height;
    int fx;
    int fy;
    int ppx;
    int ppy;
    void Set(int width_, int height_, int fx_, int fy_, int ppx_, int ppy_)
    {
        width = width_;
        height = height_;
        fx = fx_;
        fy = fy_;
        ppx = ppx_;
        ppy = ppy_;
    }
}intrinsic_param;

inline void _SPTransformFromZImageToZCamera(float one_divide_fx, float one_divide_fy, float u0, float v0, float vin1, float vin2, float vin3, float &vout1, float &vout2, float &vout3);

void GetPointCloud(unsigned short *psDepth, int width, int height, intrinsic_param intrin, float *pointcloud);

void convertDepthToColor(cv::Mat& depth, cv::Mat& color);


inline void _SPTransformFromZImageToZCamera(float one_divide_fx, float one_divide_fy, float u0, float v0, float vin1, float vin2, float vin3, float &vout1, float &vout2, float &vout3)
{
    vout1 = vin3*(vin1 - u0)*one_divide_fx;
    vout2 = vin3*(vin2 - v0)*one_divide_fy;
    vout3 = vin3;
}

void GetPointCloud(unsigned short *psDepth, int width, int height, intrinsic_param intrin, float *pointcloud)
{
    float one_divide_fx = 1.0f / intrin.fx;
    float one_divide_fy = 1.0f / intrin.fy;
    float v1, v2, v3;
    float _v1, _v2, _v3;

    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            int i = y * width + x;
            float d = (float)psDepth[i];

            if (0)//d < 1000.0f || d > 10000.0f)
            {
                pointcloud[3 * i] = 0.0f;
                pointcloud[3 * i + 1] = 0.0f;
                pointcloud[3 * i + 2] = 0.0f;
            }
            else
            {
                _SPTransformFromZImageToZCamera(one_divide_fx, one_divide_fy, intrin.ppx, intrin.ppy,
                    x, y, d, v1, v2, v3);
                pointcloud[3 * i] = 0.0001*v1;
                pointcloud[3 * i + 1] = 0.0001*v2;
                pointcloud[3 * i + 2] = 0.0001*v3;
            }
        }
    }
}

void convertDepthToColor(cv::Mat& depth, cv::Mat& color)
{
    const uint8_t nearColor[] = { 255, 0, 0 }, farColor[] = { 20, 40, 255 };
    auto rgbImage = color.ptr<uint8_t>();
    auto depthImage = depth.ptr<uint16_t>();
    auto width = color.cols;
    auto height = color.rows;
    // Produce a cumulative histogram of depth values
    int histogram[256 * 256] = { 1 };
    for (int i = 0; i < width * height; ++i)
    {
        if (auto d = depthImage[i]) ++histogram[d];
    }
    for (int i = 1; i < 256 * 256; i++)
    {
        histogram[i] += histogram[i - 1];
    }
    // Remap the cumulative histogram to the range 0..256
    for (int i = 1; i < 256 * 256; i++)
    {
        histogram[i] = (histogram[i] << 8) / histogram[256 * 256 - 1];
    }
    // Produce RGB image by using the histogram to interpolate between two colors
    auto rgb = rgbImage;
    for (int i = 0; i < width * height; i++)
    {
        if (uint16_t d = depthImage[i]) // For valid depth values (depth > 0)
        {
            auto t = histogram[d]; // Use the histogram entry (in the range of 0..256) to interpolate between nearColor and farColor
            *rgb++ = ((256 - t) * nearColor[0] + t * farColor[0]) >> 8;
            *rgb++ = ((256 - t) * nearColor[1] + t * farColor[1]) >> 8;
            *rgb++ = ((256 - t) * nearColor[2] + t * farColor[2]) >> 8;
        }
        else // Use black pixels for invalid values (depth == 0)
        {
            *rgb++ = 0;
            *rgb++ = 0;
            *rgb++ = 0;
        }                //show color image converted from depth
    }//end for
}
