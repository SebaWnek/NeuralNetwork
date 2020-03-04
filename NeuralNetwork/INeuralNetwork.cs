using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>
    /// Neural network interface
    /// </summary>
    interface INeuralNetwork
    {
        void CalculateNetwork(double[] inputs);
        double[] GetOutputs();
        void TrainNetwork(double[] inputs, double[] desiredOutput);
        void UpdateLearningRate(double learningRate);
    }
}
