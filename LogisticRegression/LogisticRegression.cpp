// LogisticRegression.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


inline CNTK::FunctionPtr FullyConnectedLinearLayer(CNTK::Variable input, size_t outputDim, 
	const CNTK::DeviceDescriptor& device, const std::wstring& outputName = L"")
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	CNTK::Parameter timesParam({ outputDim, inputDim }, CNTK::DataType::Float,
		//CNTK::GlorotUniformInitializer(CNTK::DefaultParamInitScale, CNTK::SentinelValueForInferParamInitRank, CNTK::SentinelValueForInferParamInitRank, 
		CNTK::NormalInitializer(0.05, CNTK::SentinelValueForInferParamInitRank, CNTK::SentinelValueForInferParamInitRank,
			std::chrono::high_resolution_clock::now().time_since_epoch().count()), device, L"hiddenWeights");
	CNTK::FunctionPtr timesFunction = CNTK::Times(timesParam, input, L"times");

	CNTK::Parameter plusParam({ outputDim }, 0.0f, device, L"hiddenBiases");
	return CNTK::Plus(plusParam, timesFunction, outputName);
}

inline CNTK::FunctionPtr FullyConnectedDNNLayer(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device,
	const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& nonLinearity, const std::wstring& outputName = L"")
{
	return nonLinearity(FullyConnectedLinearLayer(input, outputDim, device, outputName));
}


inline void PrintTrainingProgress(const CNTK::TrainerPtr trainer, size_t minibatchIdx, size_t outputFrequencyInMinibatches)
{
	if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer->PreviousMinibatchSampleCount() != 0)
	{
		double trainLossValue = trainer->PreviousMinibatchLossAverage();
		double evaluationValue = trainer->PreviousMinibatchEvaluationAverage();
		printf("Minibatch %d: Training loss = %.8g, Evaluation = %.8g\n", (int)minibatchIdx, trainLossValue, evaluationValue);
	}
}

int main()
{
	CNTK::DeviceDescriptor gpuDevice = CNTK::DeviceDescriptor::GPUDevice(0);
	std::wstring deviceString = gpuDevice.AsString();
	CNTK::GPUProperties gpuProps = CNTK::DeviceDescriptor::GetGPUProperties(gpuDevice);

	std::wcout << L"GPU device 0:" << std::endl;
	std::wcout << deviceString << std::endl;

	const size_t inputDim = 2;
	const size_t outputDim = 1;
	const size_t hiddenLayerDim_1 = 20;
	const size_t hiddenLayerDim_2 = 20;
	const double learningRate = 0.1;

	const wchar_t *training_file = L"Train_cntk_text.txt";
	const wchar_t *testing_file = L"Test_cntk_text.txt";
	const wchar_t *featureStreamName = L"features";
	const wchar_t *labelsStreamName = L"labels";

	CNTK::Variable input = CNTK::InputVariable({ inputDim }, CNTK::DataType::Float, L"input_features");
	CNTK::FunctionPtr scaledInput = CNTK::Plus(input, CNTK::Constant::Scalar(0.5f));
	scaledInput = CNTK::ElementTimes(scaledInput, CNTK::Constant::Scalar(0.15f), L"scaled_input");
	
	CNTK::FunctionPtr hidden1 = FullyConnectedDNNLayer(scaledInput, hiddenLayerDim_1, gpuDevice,
		std::bind(CNTK::Sigmoid, _1, L"activation"), L"hidden1");

	CNTK::FunctionPtr hidden2 = FullyConnectedDNNLayer(hidden1, hiddenLayerDim_2, gpuDevice,
		std::bind(CNTK::Sigmoid, _1, L"activation"), L"hidden2");

	CNTK::FunctionPtr output = FullyConnectedDNNLayer(hidden2, outputDim, gpuDevice, std::bind(CNTK::Sigmoid, _1, L"activation"), L"output");

	CNTK::Variable target = CNTK::InputVariable({ outputDim }, CNTK::DataType::Float, L"target");
	//CNTK::FunctionPtr scaledTarget = CNTK::Plus(target, CNTK::Constant::Scalar(0.5f));
	//scaledTarget = CNTK::ElementTimes(scaledTarget, CNTK::Constant::Scalar(0.15f), L"scaled_target");

	CNTK::FunctionPtr trainingLoss = CNTK::CrossEntropyWithSoftmax(output, target, L"loss_func");
	CNTK::FunctionPtr evaluation = CNTK::ClassificationError(output, target, L"classification_error");

	const size_t minibatchSize = 64;
	const size_t numSamplesPerSweep = 1000;
	const size_t numSweepsToTrainWith = 40;
	const size_t numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;

	CNTK::MinibatchSourcePtr mbSource = CNTK::TextFormatMinibatchSource(training_file, { {featureStreamName, inputDim}, { labelsStreamName, outputDim} });
	CNTK::StreamInformation featureStreamInfo = mbSource->StreamInfo(featureStreamName);
	CNTK::StreamInformation labelStreamInfo = mbSource->StreamInfo(labelsStreamName);

	CNTK::LearningRateSchedule learningRatePerSample = CNTK::TrainingParameterPerSampleSchedule(learningRate);

	CNTK::TrainerPtr trainer = CNTK::CreateTrainer(output, trainingLoss, evaluation, 
	{ CNTK::SGDLearner(output->Parameters(), learningRatePerSample) }, 
	{} /*TODO: Progress witers*/);

	size_t outputFrequencyInMinibatches = 20;

	for (size_t i = 0; i < numMinibatchesToTrain; ++i)
	{
		std::unordered_map<CNTK::StreamInformation, CNTK::MinibatchData> mbData = mbSource->GetNextMinibatch(minibatchSize, gpuDevice);

		trainer->TrainMinibatch({ {input, mbData[featureStreamInfo]}, {target, mbData[labelStreamInfo]} }, gpuDevice);

		PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
	}

	if (IsDebuggerPresent())
		system("pause");

    return 0;
}

