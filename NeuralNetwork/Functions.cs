using System;

namespace NeuralNetwork
{
    static class Functions
    {
        public static float[] TanH(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (float)Math.Tanh(input[i]);
            }
            return output;
        }

        public static float[] DTanH(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = 1 - input[i] * input[i];
            }
            return output;
        }

        public static float[] LeakyReLU(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? input[i] : 0.01f * input[i];
            }
            return output;
        }

        public static float[] DLeakyReLU(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? 1 : 0.01f;
            }
            return output;
        }
        public static float[] ReLU(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? input[i] : 0;
            }
            return output;
        }

        public static float[] DReLU(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? 1 : 0;
            }
            return output;
        }

        public static float[] Sigmoid(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (float)(1 / (1 + Math.Pow(Math.E, -1 * input[i])));
            }
            return output;
        }

        public static float[] DSigmoid(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] * (1 - input[i]);
            }
            return output;
        }
        public static float[] Linear(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i];
            }
            return output;
        }

        public static float[] DLinear(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = 1;
            }
            return output;
        }


    }
}
