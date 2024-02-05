#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>

static constexpr auto Pi = 3.14159265;

inline float linearToDecibel(float amplitude)
{
  return 20.0f * std::log10(amplitude);
}

inline float decibelToLinear(float dB)
{
  return std::pow(10.0f, dB / 20.0f);
}

typedef struct
{
  double f0;      // Center frequency of the peak
  double q;       // Quality factor of the peak
  double fs;      // Sample rate
  double a, b, c; // Filter coefficients
  double x1, x2;  // Input state variables
  double y1, y2;  // Output state variables
} peak_filter_t;

auto peak_filter_init(peak_filter_t *filter, double f0, double q, double fs) -> void
{
  filter->f0 = f0;
  filter->q = q;
  filter->fs = fs;
  filter->x1 = 0;
  filter->x2 = 0;
  filter->y1 = 0;
  filter->y2 = 0;
}

auto peak_filter_process(peak_filter_t *filter, double *input, double *output,
                         uint32_t num_samples) -> void
{
  double omega = 2 * Pi * filter->f0 / filter->fs;
  double alpha = sin(omega) / (2 * filter->q);
  double a = 1 + alpha;
  filter->a = alpha / a;
  filter->b = 0;
  filter->c = -alpha / a;
  for (uint32_t i = 0; i < num_samples; ++i)
  {
    double x = input[i];
    double y = filter->a * x + filter->b * filter->x1 + filter->c * filter->x2 -
               filter->b * filter->y1 - filter->c * filter->y2;
    output[i] = y;
    filter->x2 = filter->x1;
    filter->x1 = x;
    filter->y2 = filter->y1;
    filter->y1 = y;
  }
}

auto peak_filter_response(peak_filter_t *filter, double f) -> std::complex<double>
{
  using namespace std::complex_literals;
  double omega = 2 * Pi * f / filter->fs;
  std::complex<double> z = exp(-1i * omega);
  return (1.0 + filter->a * z + filter->c * z * z) /
         (1.0 - filter->a * z + filter->c * z * z);
}

auto main() -> int
{
  auto filter = peak_filter_t{};
  peak_filter_init(&filter, 440.0, 0.71, 44'100.0);
  peak_filter_process(&filter, nullptr, nullptr, 0);
  for (auto f : {20.0, 100.0, 200.0, 400.0, 440.0, 800.0, 1000.0, 5000.0})
  {
    auto const z = peak_filter_response(&filter, f);
    std::printf("f: %f, g: %f\n", f, linearToDecibel(std::abs(z)));
  }
}