using System;

namespace NeuralNetwork
{
    /// <summary>
    /// Possible activation functions
    /// </summary>
    public enum FunctionTypes
    {
        TanH,
        Sigmoid,
        ReLU,
        LeakyReLU,
        Linear
    }

    /// <summary>
    /// Activation functions. Must contain function and it's derivative named DFunction.
    /// </summary>
    public static class Functions
    {
        /// <summary>
        /// Factory method for generating delegates to be used by layers as activation functions.
        /// </summary>
        /// <param name="type">Type of function to be provided</param>
        /// <returns>Tuble consiting of delegates of activation functuon and its derivative</returns>
        public static (Func<double[], double[]> func, Func<double[], double[]> derivative) GetFunctions(FunctionTypes type)
        {
            switch (type)
            {
                case FunctionTypes.TanH:
                    return (x => TanH(x), x => DTanH(x));
                case FunctionTypes.Sigmoid:
                    return (x => Sigmoid(x), x => DSigmoid(x));
                case FunctionTypes.ReLU:
                    return (x => ReLU(x), x => DReLU(x));
                case FunctionTypes.LeakyReLU:
                    return (x => LeakyReLU(x), x => DLeakyReLU(x));
                case FunctionTypes.Linear:
                    return (x => Linear(x), x => DLinear(x));
                default:
                    return (x => Sigmoid(x), x => DSigmoid(x));
            }
        }
        /// <summary>
        /// Hyberbolic tangent
        /// </summary>
        private static double[] TanH(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (double)Math.Tanh(input[i]);
            }
            return output;
        }

        private static double[] DTanH(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = 1 - input[i] * input[i];
            }
            return output;
        }
        /// <summary>
        /// Leaky rectified linear unit
        /// </summary>
        private static double[] LeakyReLU(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? input[i] : 0.01f * input[i];
            }
            return output;
        }

        private static double[] DLeakyReLU(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? 1 : 0.01f;
            }
            return output;
        }
        /// <summary>
        /// Rectified linear unit
        /// </summary>
        private static double[] ReLU(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? input[i] : 0;
            }
            return output;
        }

        private static double[] DReLU(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? 1 : 0;
            }
            return output;
        }
        /// <summary>
        /// Sigmoid function
        /// </summary>
        private static double[] Sigmoid(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (double)(1 / (1 + Math.Pow(Math.E, -1 * input[i])));
            }
            return output;
        }

        private static double[] DSigmoid(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] * (1 - input[i]);
            }
            return output;
        }
        /// <summary>
        /// Linear function
        /// </summary>
        private static double[] Linear(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i];
            }
            return output;
        }

        private static double[] DLinear(double[] input)
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
