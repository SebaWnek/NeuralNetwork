namespace NeuralNetwork
{
    class Input
    {
        public float[] Inputs { get; set; }

        public Input(int count)
        {
            Inputs = new float[count];
        }
    }
}
