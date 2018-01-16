// Conv2DAutoencoder.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "cntk_helper.hpp"

std::wstring training_images_path = L"C:\\Users\\Alex\\Desktop\\Datasets\\MNIST Handwritten Characters\\train-images.idx3-ubyte";
std::wstring testing_images_path = L"C:\\Users\\Alex\\Desktop\\Datasets\\MNIST Handwritten Characters\\t10k-images.idx3-ubyte";
std::wstring training_labels_path = L"C:\\Users\\Alex\\Desktop\\Datasets\\MNIST Handwritten Characters\\train-labels.idx1-ubyte";
std::wstring testing_labels_path = L"C:\\Users\\Alex\\Desktop\\Datasets\\MNIST Handwritten Characters\\t10k-labels.idx1-ubyte";

constexpr size_t image_height = 28;
constexpr size_t image_width = 28;
constexpr size_t num_training_images = 60000;
constexpr size_t num_testing_images = 10000;

std::unique_ptr<float[]> trainingImageData;
std::unique_ptr<float[]> testingImageData;

std::vector<CNTK::NDArrayViewPtr> trainingImageArrays;
std::vector<CNTK::NDArrayViewPtr> testingImageArrays;

CNTK::Variable inputVar, targetVar;
CNTK::FunctionPtr encoder, decoder, trainingDecoder;
CNTK::Variable decoderInput;
CNTK::TrainerPtr trainer;

bool LoadBinaryData(std::wstring path, size_t offset, size_t data_size, unsigned char *buf);
bool LoadMNIST();
float *GetTrainingImagePtr(size_t image_index);
float *GetTestingImagePtr(size_t image_index);
void CreateModel();
void CreateModel_TiedWeights();
void LoadModelFromFile();
void CreateTrainer();
void Train();
void DisplayResult();

int main()
{
	if (!LoadMNIST())
		return 1;

	const bool training = false;

	if (training)
	{
		CreateModel();
		CreateTrainer();
		Train();
	}
	else
	{
		LoadModelFromFile();
	}

	DisplayResult();

	/*CreateModel_TiedWeights();
	CreateTrainer();
	Train();
	DisplayResult();*/


	return 0;
}

void LoadModelFromFile()
{
	trainingDecoder = CNTK::Function::Load(L"autoencoder.cntkmodel", CNTK::DeviceDescriptor::GPUDevice(0));
	for (auto input : trainingDecoder->Inputs())
	{
		if (input.Name() == L"encoder/input")
		{
			inputVar = input;
			break;
		}
		//std::wcout << L"Input: " << input.Name() << std::endl;
	}
}


bool LoadBinaryData(std::wstring path, size_t offset, size_t data_size, unsigned char *buf)
{
	std::ifstream fin(path, std::ios::binary);

	if (!fin.is_open())
	{
		std::wcerr << L"Unable to open: " << path << std::endl;
		return false;
	}

	fin.seekg(offset, std::ios::beg);

	fin.read(reinterpret_cast<char*>(buf), data_size);

	return true;
}

float *GetTrainingImagePtr(size_t image_index)
{
	return trainingImageData.get() + (image_index * image_height * image_width);
}

float *GetTestingImagePtr(size_t image_index)
{
	return testingImageData.get() + (image_index * image_height * image_width);
}


bool LoadMNIST()
{
	const size_t training_images_size = num_training_images * image_height * image_width;
	const size_t testing_images_size = num_testing_images * image_height * image_width;
	const size_t training_labels_size = num_training_images;
	const size_t testing_labels_size = num_testing_images;
	const size_t image_file_offset = sizeof(int) * 4;
	const size_t label_file_offset = sizeof(int) * 2;

	std::unique_ptr<unsigned char[]> rawTrainingImageData(new unsigned char[training_images_size]);
	std::unique_ptr<unsigned char[]> rawTestingImageData(new unsigned char[testing_images_size]);
	std::unique_ptr<unsigned char[]> rawTrainingLabelData(new unsigned char[training_labels_size]);
	std::unique_ptr<unsigned char[]> rawTestingLabelData(new unsigned char[testing_labels_size]);

	if (!(
		LoadBinaryData(training_images_path, image_file_offset, training_images_size, rawTrainingImageData.get()) &&
		LoadBinaryData(testing_images_path, image_file_offset, testing_images_size, rawTestingImageData.get()) &&
		LoadBinaryData(training_labels_path, label_file_offset, training_labels_size, rawTrainingLabelData.get()) &&
		LoadBinaryData(testing_labels_path, label_file_offset, testing_labels_size, rawTestingLabelData.get())
		))
	{
		std::wcerr << "Failed to load MNIST!" << std::endl;
		return false;
	}

	trainingImageData.reset(new float[training_images_size]);
	testingImageData.reset(new float[testing_images_size]);

	for (size_t i = 0; i < training_images_size; i++)
		trainingImageData[i] = rawTrainingImageData[i] / 255.0f;

	for (size_t i = 0; i < testing_images_size; i++)
		testingImageData[i] = rawTestingImageData[i] / 255.0f;

	CNTK::DeviceDescriptor cpu = CNTK::DeviceDescriptor::CPUDevice();

	CNTK::NDShape sampleShape = { image_height , image_width, 1 };
	CNTK::NDShape sequenceShape = { image_height , image_width, 1, 1 };
	const size_t sampleSize = sampleShape.TotalSize();

	trainingImageArrays.resize(num_training_images);
	testingImageArrays.resize(num_testing_images);

	for (size_t i = 0; i < num_training_images; i++)
		trainingImageArrays[i] = CNTK::MakeSharedObject<CNTK::NDArrayView>(sequenceShape, GetTrainingImagePtr(i), sampleSize, cpu);

	for (size_t i = 0; i < num_testing_images; i++)
		testingImageArrays[i] = CNTK::MakeSharedObject<CNTK::NDArrayView>(sequenceShape, GetTestingImagePtr(i), sampleSize, cpu);

	return true;
}

void CreateModel()
{
	CNTK::DeviceDescriptor gpu = CNTK::DeviceDescriptor::GPUDevice(0);

	CNTK::NDShape imageShape = { image_height , image_width, 1 };

	inputVar = CNTK::InputVariable(imageShape, CNTK::DataType::Float, L"encoder/input");

	encoder = Conv2DLayer(gpu, inputVar, 64, 7, 7, true, 1, 1, 0.26, L"encoder/convlayer1");
	encoder = CNTK::ReLU(encoder, L"encoder/convlayer1/activation");
	encoder = Conv2DLayer(gpu, encoder, 64, 5, 5, true, 1, 1, 0.26, L"encoder/convlayer2");
	encoder = CNTK::ReLU(encoder, L"encoder/convlayer2/activation");

	decoderInput = CNTK::PlaceholderVariable(encoder->Output().Shape(), L"decoder/input", CNTK::Axis::DefaultInputVariableDynamicAxes());
	//decoderInput = encoder;

	std::wcout << encoder->Output().Shape().AsString() << std::endl;

	decoder = ConvTranspose2DLayer(gpu, decoderInput, 64, 5, 5, true, 1, 1, 0.26, L"decoder/deconvlayer1");
	decoder = CNTK::ReLU(decoder, L"decoder/deconvlayer1/activation");
	std::wcout << decoder->Output().Shape().AsString() << std::endl;
	decoder = Conv2DLayer(gpu, decoder, 1, 7, 7, true, 1, 1, 0.26, L"decoder/deconvlayer2");
	decoder = CNTK::ReLU(decoder, L"decoder/deconvlayer2/activation");
	std::wcout << decoder->Output().Shape().AsString() << std::endl;

	trainingDecoder = decoder->Clone(CNTK::ParameterCloningMethod::Share, { { decoderInput, encoder } });
}

void CreateModel_TiedWeights()
{
	CNTK::DeviceDescriptor gpu = CNTK::DeviceDescriptor::GPUDevice(0);

	CNTK::NDShape imageShape = { image_height * image_width };

	inputVar = CNTK::InputVariable(imageShape, CNTK::DataType::Float, L"encoder/input");

	CNTK::Parameter weight1 = NewWeightParameter({ 400, imageShape.TotalSize(), }, gpu, L"shared/weight1");
	CNTK::Parameter bias1 = NewBiasParameter({ 400 }, gpu, L"shared/bias1");
	CNTK::Parameter weight2 = NewWeightParameter({ 40, 400 }, gpu, L"shared/weight2");
	CNTK::Parameter bias2 = NewBiasParameter({ 40 }, gpu, L"shared/bias2");
	CNTK::Parameter bias3 = NewBiasParameter({ 400 }, gpu, L"decoder/hidden1/bias");
	CNTK::Parameter bias4 = NewBiasParameter(imageShape, gpu, L"decoder/hidden2/bias");

	encoder = FullyConnectedDNNLayer_SharedWeight(inputVar, weight1, bias1, CNTK::Sigmoid, L"encoder/hidden1");
	encoder = FullyConnectedDNNLayer_SharedWeight(encoder, weight2, bias2, CNTK::Sigmoid, L"encoder/output");

	decoder = FullyConnectedDNNLayer_SharedWeight(encoder, CNTK::TransposeAxes(weight2, CNTK::Axis(0), CNTK::Axis(1)), bias3, CNTK::Sigmoid, L"decoder/hidden1");
	decoder = FullyConnectedDNNLayer_SharedWeight(decoder, CNTK::TransposeAxes(weight1, CNTK::Axis(0), CNTK::Axis(1)), bias4, CNTK::Sigmoid, L"decoder/output");

}

void CreateTrainer()
{
	CNTK::DeviceDescriptor gpu = CNTK::DeviceDescriptor::GPUDevice(0);

	targetVar = CNTK::InputVariable({ image_height , image_width, 1 }, CNTK::DataType::Float, L"decoder/target");

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

	const size_t num_epochs = 10;
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
				arrays.push_back(trainingImageArrays[indices[batch_i * minibatch_size + i]]);
			}

			static const CNTK::NDShape imageShape = { image_height , image_width, 1 };

			CNTK::ValuePtr input_value = CNTK::Value::Create(imageShape, arrays, {}, gpu, true, true);

			trainer->TrainMinibatch({ { inputVar, input_value },{ targetVar, input_value } }, gpu);

			epoch_loss += trainer->PreviousMinibatchLossAverage();
		}

		epoch_loss /= num_minibatches;

		wprintf_s(L"Epoch %d: avg loss = %lf\n", epoch_i, epoch_loss);

	}//end of epoch loop

	trainingDecoder->Save(L"autoencoder.cntkmodel");
}

void DisplayResult()
{
	CNTK::DeviceDescriptor cpu = CNTK::DeviceDescriptor::CPUDevice();
	CNTK::DeviceDescriptor gpu = CNTK::DeviceDescriptor::GPUDevice(0);

	cv::namedWindow("input_window", cv::WindowFlags::WINDOW_NORMAL);
	cv::resizeWindow("input_window", 200, 200);
	cv::moveWindow("input_window", 200, 400);
	cv::namedWindow("output_window", cv::WindowFlags::WINDOW_NORMAL);
	cv::resizeWindow("output_window", 200, 200);
	cv::moveWindow("output_window", 400, 400);

	//CNTK::NDShape input_shape = { image_height , image_width, 1 };

	/*CNTK::NDArrayViewPtr input_image = CNTK::MakeSharedObject<CNTK::NDArrayView>
		(CNTK::DataType::Float, input_shape, GetTestingImagePtr(0), image_height * image_width * sizeof(float), gpu);
	CNTK::ValuePtr input_value = CNTK::MakeSharedObject<CNTK::Value>(input_image);*/

	for (int i = 0; i < 10; i++)
	{
		CNTK::ValuePtr input_value = CNTK::MakeSharedObject<CNTK::Value>(testingImageArrays[i]);

		std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = { { trainingDecoder, nullptr } };
		std::vector<std::vector<float>> output_floats;

		trainingDecoder->Evaluate({ { inputVar, input_value } }, outputs, gpu);

		outputs[trainingDecoder]->CopyVariableValueTo(trainingDecoder, output_floats);

		cv::Mat inputMat(image_height, image_width, CV_32FC1, GetTestingImagePtr(i));
		cv::Mat outputMat(image_height, image_width, CV_32FC1, output_floats[0].data());

		cv::imshow("input_window", inputMat);
		cv::imshow("output_window", outputMat);
		cv::waitKey();
	}

	/*std::vector<size_t> prehots = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

	CNTK::Variable decoderTestVar = CNTK::InputVariable({ 10 }, CNTK::DataType::Float);

	CNTK::FunctionPtr testingDecoder = decoder->Clone(CNTK::ParameterCloningMethod::Share, { { decoderInput, decoderTestVar } });

	//decoder->ReplacePlaceholders({ {decoderInput, decoderTestVar} });

	CNTK::ValuePtr testing_values = CNTK::Value::CreateBatch<float>(10, prehots, gpu, true);

	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs10 = { { testingDecoder, nullptr } };

	testingDecoder->Evaluate({ { decoderTestVar, testing_values } }, outputs10, cpu);

	std::vector<std::vector<float>> output_floats_10;

	outputs10[testingDecoder]->CopyVariableValueTo(testingDecoder, output_floats_10);

	for (int i = 0; i < 10; i++)
	{
		cv::Mat testingMat(image_height, image_width, CV_32FC1, output_floats_10[i].data());

		cv::imshow("input_window", testingMat);
		cv::waitKey();
	}*/

}