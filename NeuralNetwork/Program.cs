using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            //Generate network
            Random random = new Random();
            Network network = new Network(2, 1000, 1);
            //Training network
            float[] res;
            float[] inputs;
            Console.WriteLine("0%");
            int count = 1000;
            int percent = count / 100;
            for(int i = 0; i < count; i++)
            {
                inputs = new float[] { random.Next(-100, 100), random.Next(-100, 100) };
                res = new float[1];
                res[0] = inputs[0] * inputs[1];
                network.TrainNetwork(inputs, res);
                if(i%percent == 0)
                {
                    Console.Clear();
                    Console.WriteLine(i/percent + "%");
                }
            }
            //Test network
            string entry = "";
            bool correct = false;
            float[] xy = new float[2];

            //((MainWindow)plot).PrintGraph(x => x * x);

            while(entry != "Q")
            {
                Console.WriteLine("X:");
                entry = Console.ReadLine();
                correct = float.TryParse(entry, out xy[0]);

                Console.WriteLine("Y:");
                entry = Console.ReadLine();
                correct = float.TryParse(entry, out xy[1]);

                if (correct)
                {
                    network.CalculateNetwork(xy);
                    Console.WriteLine(Math.Round(network.GetOutputs()[0]));
                }
            }


            //for (int k = 0; k < 50; k++)
            //{
            //    Random random = new Random();
            //    Network network = new Network(4, k, 1);
            //    network.Layers[network.Layers.Length - 1].Function = Functions.LeakyReLU;
            //    network.Layers[network.Layers.Length - 1].Derivative = Functions.DLeakyReLU;
            //    for (int i = 0; i < 20000; i++)
            //    {
            //        int res;
            //        float[] inputs;
            //        do
            //        {
            //            inputs = new float[] { random.Next(1000), random.Next(-200, 200), random.Next(20, 70), random.Next(-5, 6) };
            //            res = (int)TestFunctions.CalculatePowerFromAngle(inputs[0], inputs[1], inputs[2], inputs[3]);

            //        } while (res == -1);
            //        float[] output = new float[1];
            //        inputs[0] = inputs[0] / 1000;
            //        inputs[1] = inputs[1] / 1000;
            //        inputs[2] = inputs[2] / 100;
            //        inputs[3] = inputs[3] / 5;
            //        output[0] = res;
            //        //Console.WriteLine(inputs[0] + " * " + inputs[1] + " = " + res);
            //        //foreach (float j in output) Console.Write(j + ", ");
            //        //Console.WriteLine();
            //        network.TrainNetwork(inputs, output);
            //    }
            //    int count = 0;
            //    double sum = 0;
            //    for (int i = 0; i < 100000; i++)
            //    {
            //        float[] result;
            //        float[] input = new float[] { random.Next(1000), random.Next(-200, 200), random.Next(20, 70), random.Next(-5, 6) };
            //        int desiredResult = (int)Math.Round(TestFunctions.CalculatePowerFromAngle(input[0], input[1], input[2], input[3]));
            //        input[0] = input[0] / 1000;
            //        input[1] = input[1] / 1000;
            //        input[2] = input[2] / 100;
            //        input[3] = input[3] / 5;
            //        network.CalculateNetwork(input);
            //        result = network.GetOutputs();
            //        if (desiredResult != -1)
            //        {
            //            count++;
            //            sum += Math.Pow((result[0] - desiredResult), 2);
            //        }
            //        //Console.Write("x: " + input[0] + ", y: " + input[1] + ", a: " + input[2] + ", w: " + input[3] + " = ");
            //        //foreach (float j in result) Console.Write((j) + " => ");
            //        //Console.WriteLine(desiredResult + " == " + ((int)Math.Round(result[0]) == desiredResult));
            //        //Console.WriteLine();
            //    }
            //    Console.WriteLine(k + ": " + count + " => " + (sum / count)); 
            //}
        }
    }
}
