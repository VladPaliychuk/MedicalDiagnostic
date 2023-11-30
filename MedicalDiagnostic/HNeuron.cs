using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalDiagnostic
{
    public class HNeuron
    {
        private double[,] weights;
        private double bias;
        private double error;
        private double learningRate = 0.3;
        public HNeuron(int size)
        {
            weights = new double[size,3];
            Initialize();
        }

        private void Initialize()
        {
            Random random = new Random();
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    weights[i ,j] = random.NextDouble();
                }
            }

            bias = random.NextDouble();
        }

        public double[,] Weights { get { return weights; } }
        public double Error { get { return error; } }

        public void UpdateWeights(double[] input)
        {
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    weights[i, j] += learningRate * error * input[i];
                }
            }
        }

        public void UpdateBiases()
        {
            bias += learningRate * error;
        }

        static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public double CalculateOutput(double[] input)
        {
            double sum = 0;
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < 3; j++)
                    sum += weights[i,j] * input[i];
            }
            sum += bias;

            return Sigmoid(sum);
        }

        public void CalculateError(double target, double output)
        {
            error = output * (1 - output) * (target - output);
        }
        public void BackpropagateError(double y, double[] outputErrors)
        {
            double weightedError = 0;
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < 3; j++)
                    weightedError += outputErrors[j] * weights[i,j];
            }

            error = Derivative(y) * weightedError;
        }
        private double Derivative(double x)
        {
            return x * (1 - x);
        }
    }
}
