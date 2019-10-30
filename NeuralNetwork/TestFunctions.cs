using System;

namespace NeuralNetwork
{
    static class TestFunctions
    {
        static public double CalculatePowerFromAngle(double X, double Y, double angle, double Wind)
        {
            double angleRad = angle * Math.PI / 180;
            double sin = Math.Sin(angleRad);
            double cos = Math.Cos(angleRad);
            double x = Math.Abs(X);
            double y = Y;
            double g = 9.81;
            double wind = Wind;
            double a = X / wind >= 0 ? Math.Abs(wind) : -Math.Abs(wind);
            double sqrt2 = Math.Sqrt(2);
            double part1 = sqrt2 * (x * sin - y * cos) * (y * a + x * g) * (a * sin + g * cos);
            double part2 = Math.Sqrt(1 / ((x * sin - y * cos) * (a * sin + g * cos)));
            double part3 = 2 * x * a * sin * sin - 2 * y * g * cos * cos - 2 * y * a * cos * sin + 2 * x * g * cos * sin;
            double power = part1 * part2 / part3;
            if (Double.IsNaN(power)) power = -1;
            return power;
        }
    }
}
