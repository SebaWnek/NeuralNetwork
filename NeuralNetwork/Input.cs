namespace NeuralNetwork
{
    public class Input
    {
        public double[] Inputs { get; set; }

        public Input(int count)
        {
            Inputs = new double[count];
        }
    }
}
