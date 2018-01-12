// Autoenconder_Geometry2D.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "cntk_helper.hpp"

constexpr size_t image_height = 800;
constexpr size_t image_width = 800;
constexpr size_t num_training_images = 1000;

constexpr size_t points_on_circle = 144;
constexpr size_t points_on_arc = 9;
constexpr size_t input_feature_size = points_on_arc * 2;

constexpr size_t testing_arc_stride = 3;
constexpr size_t num_testing_images = points_on_circle / testing_arc_stride;

std::unique_ptr<float[]> trainingData;
std::unique_ptr<float[]> testingData;

std::vector<CNTK::NDArrayViewPtr> trainingDataArrays;
std::vector<CNTK::NDArrayViewPtr> testingDataArrays;

CNTK::Variable inputVar, targetVar;
CNTK::FunctionPtr encoder, decoder, trainingDecoder;
CNTK::Variable decoderInput;
CNTK::TrainerPtr trainer;

void CreateData_Circle();
void CreateData_CircleArc();
void CreateData_Square();
void CreateData_SquareArc();
void CreateModel();
void CreateTrainer();
void Train();
void DisplayResult();


int main()
{
	CreateData_SquareArc();

	CreateModel();
	CreateTrainer();
	Train();
	DisplayResult();

    return 0;
}

float *GetTrainingDataPtr(size_t image_index)
{
	return trainingData.get() + (image_index * input_feature_size);
}

float *GetTestingDataPtr(size_t image_index)
{
	return testingData.get() + (image_index * input_feature_size);
}

void CreateData_Circle()
{
	const CNTK::DeviceDescriptor cpu = CNTK::DeviceDescriptor::CPUDevice();

	std::ranlux48 rand_engine{ std::random_device{}() };

	const float noiseAmplitude = 0.05f;
	constexpr float circleMinRadius = 0.1f;
	constexpr float circleMaxRadius = 3.0f;
	std::uniform_real_distribution<float> noiseDist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> centerDist(-2.0f, 2.0f);
	std::uniform_real_distribution<float> radiusDist(circleMinRadius, circleMaxRadius);

	trainingData.reset(new float[num_training_images * input_feature_size]);
	testingData.reset(new float[1 * input_feature_size]);

	for (size_t circle_i = 0; circle_i < num_training_images; circle_i++)
	{
		const float circleCenter_x = centerDist(rand_engine);
		const float circleCenter_y = centerDist(rand_engine);
		const float circleRadius = radiusDist(rand_engine);

		for (size_t i = 0; i < points_on_circle; i++)
		{
			const float theta = i * 2 * M_PI / points_on_circle;
			const float cosTheta = cosf(theta);
			const float sinTheta = sinf(theta);

			trainingData[circle_i * input_feature_size + i * 2 + 0] = circleCenter_x + circleRadius * cosTheta;
			trainingData[circle_i * input_feature_size + i * 2 + 1] = circleCenter_y + circleRadius * sinTheta;
		}
	}

	const float noisyCircleRadius = radiusDist(rand_engine);
	const float noisyCircleCenter_x = centerDist(rand_engine);
	const float noisyCircleCenter_y = centerDist(rand_engine);

	for (size_t i = 0; i < points_on_circle; i++)
	{
		const float theta = i * 2 * M_PI / points_on_circle;
		const float cosTheta = cosf(theta);
		const float sinTheta = sinf(theta);
		const float noisyRadius = noisyCircleRadius + noiseAmplitude * noisyCircleRadius * noiseDist(rand_engine);

		testingData[i * 2 + 0] = noisyCircleCenter_x + noisyRadius * cosTheta;
		testingData[i * 2 + 1] = noisyCircleCenter_y + noisyRadius * sinTheta;
	}

	CNTK::NDShape sampleShape = { input_feature_size };
	CNTK::NDShape sequenceShape = { input_feature_size, 1 };
	const size_t sampleSize = sampleShape.TotalSize();

	trainingDataArrays.resize(num_training_images);
	testingDataArrays.resize(1);

	for (size_t i = 0; i < num_training_images; i++)
		trainingDataArrays[i] = CNTK::MakeSharedObject<CNTK::NDArrayView>(sequenceShape, GetTrainingDataPtr(i), sampleSize, cpu);

	testingDataArrays[0] = CNTK::MakeSharedObject<CNTK::NDArrayView>(sequenceShape, GetTestingDataPtr(0), sampleSize, cpu);
}



void CreateData_CircleArc()
{
	const CNTK::DeviceDescriptor cpu = CNTK::DeviceDescriptor::CPUDevice();

	std::ranlux48 rand_engine{ std::random_device{}() };

	const float noiseAmplitude = 0.05f;
	constexpr float circleMinRadius = 0.1f;
	constexpr float circleMaxRadius = 3.0f;
	std::uniform_real_distribution<float> noiseDist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> centerDist(-2.0f, 2.0f);
	std::uniform_real_distribution<float> radiusDist(circleMinRadius, circleMaxRadius);
	std::uniform_int_distribution<int> startPointDist(0, points_on_circle - 1);

	trainingData.reset(new float[num_training_images * input_feature_size]);
	testingData.reset(new float[num_testing_images * input_feature_size]);

	for (size_t circle_i = 0; circle_i < num_training_images; circle_i++)
	{
		const float circleCenter_x = centerDist(rand_engine);
		const float circleCenter_y = centerDist(rand_engine);
		const float circleRadius = radiusDist(rand_engine);

		int startPoint = startPointDist(rand_engine);

		for (size_t i = 0; i < points_on_arc; i++)
		{
			int actual_i = (startPoint + i) % points_on_circle;
			const float theta = actual_i * 2 * M_PI / points_on_circle;
			const float cosTheta = cosf(theta);
			const float sinTheta = sinf(theta);

			trainingData[circle_i * input_feature_size + i * 2 + 0] = circleCenter_x + circleRadius * cosTheta;
			trainingData[circle_i * input_feature_size + i * 2 + 1] = circleCenter_y + circleRadius * sinTheta;
		}
	}

	const float noisyCircleRadius = radiusDist(rand_engine);
	const float noisyCircleCenter_x = centerDist(rand_engine);
	const float noisyCircleCenter_y = centerDist(rand_engine);
	//const int noisyCircleStartPoint = startPointDist(rand_engine);

	float noisy_circle_x[points_on_circle];
	float noisy_circle_y[points_on_circle];

	for (size_t i = 0; i < points_on_circle; i++)
	{
		const float theta = i * 2 * M_PI / points_on_circle;
		const float cosTheta = cosf(theta);
		const float sinTheta = sinf(theta);
		const float noisyRadius = noisyCircleRadius + noiseAmplitude * noisyCircleRadius * noiseDist(rand_engine);

		noisy_circle_x[i] = noisyCircleCenter_x + noisyRadius * cosTheta;
		noisy_circle_y[i] = noisyCircleCenter_y + noisyRadius * sinTheta;
	}

	for (size_t circle_i = 0; circle_i < num_testing_images; circle_i++)
	{
		const int noisyCircleStartPoint = circle_i * testing_arc_stride;

		for (size_t i = 0; i < points_on_arc; i++)
		{
			int actual_i = (noisyCircleStartPoint + i) % points_on_circle;

			testingData[circle_i * input_feature_size + i * 2 + 0] = noisy_circle_x[actual_i];
			testingData[circle_i * input_feature_size + i * 2 + 1] = noisy_circle_y[actual_i];
		}
	}

	CNTK::NDShape sampleShape = { input_feature_size };
	CNTK::NDShape sequenceShape = { input_feature_size, 1 };
	const size_t sampleSize = sampleShape.TotalSize();

	trainingDataArrays.resize(num_training_images);
	testingDataArrays.resize(num_testing_images);

	for (size_t i = 0; i < num_training_images; i++)
		trainingDataArrays[i] = CNTK::MakeSharedObject<CNTK::NDArrayView>(sequenceShape, GetTrainingDataPtr(i), sampleSize, cpu);

	for (size_t i = 0; i < num_testing_images; i++)
		testingDataArrays[i] = CNTK::MakeSharedObject<CNTK::NDArrayView>(sequenceShape, GetTestingDataPtr(i), sampleSize, cpu);
}

void CreateData_Square()
{
	const CNTK::DeviceDescriptor cpu = CNTK::DeviceDescriptor::CPUDevice();

	std::ranlux48 rand_engine{ std::random_device{}() };

	const float noiseAmplitude = 0.05f;
	constexpr float circleMinRadius = 0.1f;
	constexpr float circleMaxRadius = 2.0f;
	std::uniform_real_distribution<float> noiseDist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> centerDist(-1.5f, 1.5f);
	std::uniform_real_distribution<float> radiusDist(circleMinRadius, circleMaxRadius);

	trainingData.reset(new float[num_training_images * input_feature_size]);
	testingData.reset(new float[1 * input_feature_size]);

	for (size_t circle_i = 0; circle_i < num_training_images; circle_i++)
	{
		const float circleCenter_x = centerDist(rand_engine);
		const float circleCenter_y = centerDist(rand_engine);
		const float circleRadius = radiusDist(rand_engine);

		for (size_t i = 0; i < points_on_circle; i++)
		{
			const float disp = ((i % (points_on_circle / 4))* circleRadius * 2) / (points_on_circle / 4);

			float& pointX = trainingData[circle_i * input_feature_size + i * 2 + 0];
			float& pointY = trainingData[circle_i * input_feature_size + i * 2 + 1];

			if (i < points_on_circle / 4)
			{
				pointX = circleCenter_x - circleRadius + disp;
				pointY = circleCenter_y - circleRadius;
			}
			else if (i < points_on_circle / 2)
			{
				pointX = circleCenter_x + circleRadius;
				pointY = circleCenter_y - circleRadius + disp;
			}
			else if (i < (points_on_circle * 3) / 4)
			{
				pointX = circleCenter_x + circleRadius - disp;
				pointY = circleCenter_y + circleRadius;
			}
			else
			{
				pointX = circleCenter_x - circleRadius;
				pointY = circleCenter_y + circleRadius - disp;
			}
		}
	}

	const float noisyCircleRadius = radiusDist(rand_engine);
	const float noisyCircleCenter_x = centerDist(rand_engine);
	const float noisyCircleCenter_y = centerDist(rand_engine);

	for (size_t i = 0; i < points_on_circle; i++)
	{
		const float noisyRadius = noisyCircleRadius + noiseAmplitude * noisyCircleRadius * noiseDist(rand_engine);
		const float disp = ((i % (points_on_circle / 4)) * noisyCircleRadius * 2) / (points_on_circle / 4);
		float& pointX = testingData[i * 2 + 0];
		float& pointY = testingData[i * 2 + 1];

		if (i < points_on_circle / 4)
		{
			pointX = noisyCircleCenter_x - noisyCircleRadius + disp;
			pointY = noisyCircleCenter_y - noisyRadius;
		}
		else if (i < points_on_circle / 2)
		{
			pointX = noisyCircleCenter_x + noisyRadius;
			pointY = noisyCircleCenter_y - noisyCircleRadius + disp;
		}
		else if (i < (points_on_circle * 3) / 4)
		{
			pointX = noisyCircleCenter_x + noisyCircleRadius - disp;
			pointY = noisyCircleCenter_y + noisyRadius;
		}
		else
		{
			pointX = noisyCircleCenter_x - noisyRadius;
			pointY = noisyCircleCenter_y + noisyCircleRadius - disp;
		}
	}

	CNTK::NDShape sampleShape = { input_feature_size };
	CNTK::NDShape sequenceShape = { input_feature_size, 1 };
	const size_t sampleSize = sampleShape.TotalSize();

	trainingDataArrays.resize(num_training_images);
	testingDataArrays.resize(1);

	for (size_t i = 0; i < num_training_images; i++)
		trainingDataArrays[i] = CNTK::MakeSharedObject<CNTK::NDArrayView>(sequenceShape, GetTrainingDataPtr(i), sampleSize, cpu);

	testingDataArrays[0] = CNTK::MakeSharedObject<CNTK::NDArrayView>(sequenceShape, GetTestingDataPtr(0), sampleSize, cpu);
}


void CreateData_SquareArc()
{
	const CNTK::DeviceDescriptor cpu = CNTK::DeviceDescriptor::CPUDevice();

	std::ranlux48 rand_engine{ std::random_device{}() };

	const float noiseAmplitude = 0.05f;
	constexpr float circleMinRadius = 0.1f;
	constexpr float circleMaxRadius = 2.0f;
	std::uniform_real_distribution<float> noiseDist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> centerDist(-1.5f, 1.5f);
	std::uniform_real_distribution<float> radiusDist(circleMinRadius, circleMaxRadius);
	std::uniform_int_distribution<int> startPointDist(0, points_on_circle - 1);

	trainingData.reset(new float[num_training_images * input_feature_size]);
	testingData.reset(new float[num_testing_images * input_feature_size]);

	for (size_t circle_i = 0; circle_i < num_training_images; circle_i++)
	{
		const float circleCenter_x = centerDist(rand_engine);
		const float circleCenter_y = centerDist(rand_engine);
		const float circleRadius = radiusDist(rand_engine);

		int startPoint = startPointDist(rand_engine);

		for (size_t i = 0; i < points_on_arc; i++)
		{
			int actual_i = (startPoint + i) % points_on_circle;
			const float disp = ((actual_i % (points_on_circle / 4))* circleRadius * 2) / (points_on_circle / 4);

			float& pointX = trainingData[circle_i * input_feature_size + i * 2 + 0];
			float& pointY = trainingData[circle_i * input_feature_size + i * 2 + 1];

			if (actual_i < points_on_circle / 4)
			{
				pointX = circleCenter_x - circleRadius + disp;
				pointY = circleCenter_y - circleRadius;
			}
			else if (actual_i < points_on_circle / 2)
			{
				pointX = circleCenter_x + circleRadius;
				pointY = circleCenter_y - circleRadius + disp;
			}
			else if (actual_i < (points_on_circle * 3) / 4)
			{
				pointX = circleCenter_x + circleRadius - disp;
				pointY = circleCenter_y + circleRadius;
			}
			else
			{
				pointX = circleCenter_x - circleRadius;
				pointY = circleCenter_y + circleRadius - disp;
			}
		}
	}

	const float noisyCircleRadius = radiusDist(rand_engine);
	const float noisyCircleCenter_x = centerDist(rand_engine);
	const float noisyCircleCenter_y = centerDist(rand_engine); 

	float noisy_circle_x[points_on_circle];
	float noisy_circle_y[points_on_circle];

	for (size_t i = 0; i < points_on_circle; i++)
	{
		const float noisyRadius = noisyCircleRadius + noiseAmplitude * noisyCircleRadius * noiseDist(rand_engine);
		const float disp = ((i % (points_on_circle / 4)) * noisyCircleRadius * 2) / (points_on_circle / 4);
		float& pointX = noisy_circle_x[i];
		float& pointY = noisy_circle_y[i];

		if (i < points_on_circle / 4)
		{
			pointX = noisyCircleCenter_x - noisyCircleRadius + disp;
			pointY = noisyCircleCenter_y - noisyRadius;
		}
		else if (i < points_on_circle / 2)
		{
			pointX = noisyCircleCenter_x + noisyRadius;
			pointY = noisyCircleCenter_y - noisyCircleRadius + disp;
		}
		else if (i < (points_on_circle * 3) / 4)
		{
			pointX = noisyCircleCenter_x + noisyCircleRadius - disp;
			pointY = noisyCircleCenter_y + noisyRadius;
		}
		else
		{
			pointX = noisyCircleCenter_x - noisyRadius;
			pointY = noisyCircleCenter_y + noisyCircleRadius - disp;
		}
	}

	for (size_t circle_i = 0; circle_i < num_testing_images; circle_i++)
	{
		const int noisyCircleStartPoint = circle_i * testing_arc_stride;

		for (size_t i = 0; i < points_on_arc; i++)
		{
			int actual_i = (noisyCircleStartPoint + i) % points_on_circle;

			testingData[circle_i * input_feature_size + i * 2 + 0] = noisy_circle_x[actual_i];
			testingData[circle_i * input_feature_size + i * 2 + 1] = noisy_circle_y[actual_i];
		}
	}

	CNTK::NDShape sampleShape = { input_feature_size };
	CNTK::NDShape sequenceShape = { input_feature_size, 1 };
	const size_t sampleSize = sampleShape.TotalSize();

	trainingDataArrays.resize(num_training_images);
	testingDataArrays.resize(num_testing_images);

	for (size_t i = 0; i < num_training_images; i++)
		trainingDataArrays[i] = CNTK::MakeSharedObject<CNTK::NDArrayView>(sequenceShape, GetTrainingDataPtr(i), sampleSize, cpu);

	for (size_t i = 0; i < num_testing_images; i++)
		testingDataArrays[i] = CNTK::MakeSharedObject<CNTK::NDArrayView>(sequenceShape, GetTestingDataPtr(i), sampleSize, cpu);
}

void CreateModel()
{
	CNTK::DeviceDescriptor gpu = CNTK::DeviceDescriptor::GPUDevice(0);

	CNTK::NDShape imageShape = { input_feature_size };

	inputVar = CNTK::InputVariable(imageShape, CNTK::DataType::Float, L"encoder/input");

	encoder = FullyConnectedDNNLayer(inputVar, 800, gpu, CNTK::Tanh, L"encoder/hidden1");
	encoder = FullyConnectedDNNLayer(encoder, 4, gpu, CNTK::Tanh, L"encoder/output");

	decoderInput = CNTK::PlaceholderVariable(encoder->Output().Shape(), L"decoder/input", CNTK::Axis::DefaultInputVariableDynamicAxes());
	//decoderInput = encoder;

	decoder = FullyConnectedDNNLayer(decoderInput, 800, gpu, CNTK::Tanh, L"decoder/hidden1");
	decoder = FullyConnectedLinearLayer(decoder, imageShape.TotalSize(), gpu, L"decoder/output");

	trainingDecoder = decoder->Clone(CNTK::ParameterCloningMethod::Share, { { decoderInput, encoder } });
}


void CreateTrainer()
{
	CNTK::DeviceDescriptor gpu = CNTK::DeviceDescriptor::GPUDevice(0);

	targetVar = CNTK::InputVariable({ input_feature_size }, CNTK::DataType::Float, L"decoder/target");

	CNTK::FunctionPtr loss = CNTK::SquaredError(trainingDecoder, targetVar, L"loss");

	CNTK::LearningRateSchedule learningRate = CNTK::TrainingParameterPerSampleSchedule(0.0001);
	CNTK::MomentumSchedule momentum = CNTK::TrainingParameterPerSampleSchedule(0.0000);

	CNTK::LearnerPtr learner = CNTK::AdamLearner(trainingDecoder->Parameters(), learningRate, momentum);

	trainer = CNTK::CreateTrainer(trainingDecoder, loss, { learner });
}

void Train()
{
	CNTK::DeviceDescriptor cpu = CNTK::DeviceDescriptor::CPUDevice();
	CNTK::DeviceDescriptor gpu = CNTK::DeviceDescriptor::GPUDevice(0);

	//decoder->ReplacePlaceholder(encoder);

	std::ranlux48 rand_engine(std::random_device{}());

	std::vector<size_t> indices(num_training_images);
	std::iota(indices.begin(), indices.end(), 0);

	const size_t num_epochs = 10000;
	const size_t minibatch_size = 64;
	const size_t num_minibatches = (num_training_images + minibatch_size - 1) / minibatch_size;

	for (size_t epoch_i = 0; epoch_i < num_epochs; epoch_i++)
	{
		std::shuffle(indices.begin(), indices.end(), rand_engine);

		double epoch_loss = 0.0;

		for (size_t batch_i = 0; batch_i < num_minibatches; batch_i++)
		{
			std::vector<CNTK::NDArrayViewPtr> arrays;
			arrays.reserve(num_minibatches);

			for (size_t i = 0; i < minibatch_size && (batch_i * minibatch_size + i) < num_training_images; i++)
			{
				arrays.push_back(trainingDataArrays[indices[batch_i * minibatch_size + i]]);
			}

			static const CNTK::NDShape imageShape = { input_feature_size };

			CNTK::ValuePtr input_value = CNTK::Value::Create(imageShape, arrays, {}, gpu, true, true);

			trainer->TrainMinibatch({ { inputVar, input_value },{ targetVar, input_value } }, gpu);

			epoch_loss += trainer->PreviousMinibatchLossAverage();
		}

		epoch_loss /= num_minibatches;

		wprintf_s(L"Epoch %d: avg loss = %lf\n", epoch_i, epoch_loss);

	}//end of epoch loop
}


void DisplayResult()
{
	CNTK::DeviceDescriptor cpu = CNTK::DeviceDescriptor::CPUDevice();
	CNTK::DeviceDescriptor gpu = CNTK::DeviceDescriptor::GPUDevice(0);

	CNTK::NDShape input_shape = { num_testing_images, input_feature_size };

	/*CNTK::NDArrayViewPtr input_image = CNTK::MakeSharedObject<CNTK::NDArrayView>
		(CNTK::DataType::Float, input_shape, GetTestingDataPtr(0), input_shape.TotalSize() * sizeof(float), cpu);
	CNTK::ValuePtr input_value = CNTK::MakeSharedObject<CNTK::Value>(input_image);*/

	CNTK::ValuePtr input_value = CNTK::Value::Create({ input_feature_size }, testingDataArrays, {}, cpu, true);

	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = { { trainingDecoder, nullptr } };
	std::vector<std::vector<float>> output_floats;

	trainingDecoder->Evaluate({ { inputVar, input_value } }, outputs, cpu);

	outputs[trainingDecoder]->CopyVariableValueTo(trainingDecoder, output_floats);

	float circle_x[points_on_circle];
	float circle_y[points_on_circle];

	ZeroMemory(circle_x, sizeof(circle_x));
	ZeroMemory(circle_y, sizeof(circle_y));

	for (size_t circle_i = 0; circle_i < num_testing_images; circle_i++)
	{
		size_t startIndex = circle_i * testing_arc_stride;

		for (size_t i = 0; i < points_on_arc; i++)
		{
			size_t actual_i = (startIndex + i) % points_on_circle; 
			
			const float *original_pt1 = &output_floats[circle_i][i * 2];

			circle_x[actual_i] += original_pt1[0];
			circle_y[actual_i] += original_pt1[1];
		}
	}

	for (size_t i = 0; i < points_on_circle; i++)
	{
		circle_x[i] /= 3.0f;
		circle_y[i] /= 3.0f;
	}

	cv::Mat targetCircle(image_height, image_width, CV_8UC3);
	cv::Mat noisyCircle(image_height, image_width, CV_8UC3);

	cv::Scalar red(0.0, 0.0, 255.0, 255.0);
	cv::Scalar green(0.0, 127.0, 0.0, 255.0);
	const int circleThickness = 2;

	auto transform = [=](float x) {return (x + 3.7f) * (image_width / 7.4f); };


	for (size_t i = 0; i < points_on_circle; i++)
	{
		cv::Point2f original_point1(transform(circle_x[i]), transform(circle_y[i]));
		cv::Point2f original_point2(transform(circle_x[(i + 1) % points_on_circle]), transform(circle_y[(i + 1) % points_on_circle]));

		cv::line(targetCircle, original_point1, original_point2, red, circleThickness, CV_AA);
	}

	for (size_t circle_i = 0; circle_i < num_testing_images; circle_i++)
	{
		for (size_t i = 0; i < points_on_arc - 1; i++)
		{
			//const float *original_pt1 = &trainingData[i * 2];
			//const float *original_pt2 = &trainingData[((i + 1) % points_per_circle) * 2];
			///const float *original_pt1 = &output_floats[circle_i][i * 2];
			//const float *original_pt2 = &output_floats[circle_i][((i + 1) % points_on_arc) * 2];
			const float *noisy_pt1 = &testingData[circle_i * input_feature_size + i * 2];
			const float *noisy_pt2 = &testingData[circle_i * input_feature_size + ((i + 1) % points_on_arc) * 2];

			//cv::Point2f original_point1(transform(original_pt1[0]), transform(original_pt1[1]));
			//cv::Point2f original_point2(transform(original_pt2[0]), transform(original_pt2[1]));
			cv::Point2f noisy_point1(transform(noisy_pt1[0]), transform(noisy_pt1[1]));
			cv::Point2f noisy_point2(transform(noisy_pt2[0]), transform(noisy_pt2[1]));

			//cv::line(targetCircle, original_point1, original_point2, red, circleThickness, CV_AA);
			cv::line(noisyCircle, noisy_point1, noisy_point2, green, circleThickness, CV_AA);
		}
	}

	cv::namedWindow("input_window", cv::WindowFlags::WINDOW_NORMAL);
	cv::resizeWindow("input_window", 800, 800);
	cv::moveWindow("input_window", 200, 400);
	cv::namedWindow("output_window", cv::WindowFlags::WINDOW_NORMAL);
	cv::resizeWindow("output_window", 800, 800);
	cv::moveWindow("output_window", 1000, 400);

	cv::imshow("input_window", targetCircle);
	cv::imshow("output_window", noisyCircle);
	cv::waitKey();
}