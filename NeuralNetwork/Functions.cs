using System;

namespace NeuralNetwork
{
    public static class Functions
    {
        public static double[] TanH(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (double)Math.Tanh(input[i]);
            }
            return output;
        }

        public static double[] DTanH(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = 1 - input[i] * input[i];
            }
            return output;
        }

        public static double[] LeakyReLU(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? input[i] : 0.01f * input[i];
            }
            return output;
        }

        public static double[] DLeakyReLU(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? 1 : 0.01f;
            }
            return output;
        }
        public static double[] ReLU(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? input[i] : 0;
            }
            return output;
        }

        public static double[] DReLU(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? 1 : 0;
            }
            return output;
        }

        public static double[] Sigmoid(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (double)(1 / (1 + Math.Pow(Math.E, -1 * input[i])));
            }
            return output;
        }

        public static double[] DSigmoid(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] * (1 - input[i]);
            }
            return output;
        }
        public static double[] Linear(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i];
            }
            return output;
        }

        public static double[] DLinear(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = 1;
            }
            return output;
        }


    }
}
