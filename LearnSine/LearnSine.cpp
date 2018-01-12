// LearnSine.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

inline CNTK::FunctionPtr FullyConnectedLinearLayer(CNTK::Variable input, size_t outputDim,
	const CNTK::DeviceDescriptor& device, const std::wstring& outputName = L"")
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	CNTK::Parameter timesParam({ outputDim, inputDim }, CNTK::DataType::Float,
		CNTK::GlorotUniformInitializer(1, CNTK::SentinelValueForInferParamInitRank, CNTK::SentinelValueForInferParamInitRank,
		//CNTK::NormalInitializer(0.05, CNTK::SentinelValueForInferParamInitRank, CNTK::SentinelValueForInferParamInitRank,
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

static constexpr float pi = 3.1415f;

int main()
{
	CNTK::DeviceDescriptor gpuDevice = CNTK::DeviceDescriptor::GPUDevice(0);

	std::wstring deviceString = gpuDevice.AsString();
	CNTK::GPUProperties gpuProps = CNTK::DeviceDescriptor::GetGPUProperties(gpuDevice);

	std::wcout << L"GPU device 0:" << std::endl;
	std::wcout << deviceString << std::endl;

	const size_t numHidden_1 = 5;
	const size_t numHidden_2 = 5;
	const size_t numHidden_3 = 5;

	CNTK::Variable input = CNTK::InputVariable({ 1 }, CNTK::DataType::Float);
	CNTK::FunctionPtr normalizedInput = CNTK::ElementDivide(input, CNTK::Constant::Scalar(4 * pi));
	normalizedInput = CNTK::Plus(normalizedInput, CNTK::Constant::Scalar(0.5f));

	CNTK::FunctionPtr hidden1 = FullyConnectedDNNLayer(normalizedInput, numHidden_1, gpuDevice,
		std::bind(CNTK::LeakyReLU, _1, L"activation"), L"hidden1");

	CNTK::FunctionPtr hidden2 = FullyConnectedDNNLayer(hidden1, numHidden_2, gpuDevice,
		std::bind(CNTK::LeakyReLU, _1, L"activation"), L"hidden2");

	CNTK::FunctionPtr hidden3 = FullyConnectedDNNLayer(hidden2, numHidden_3, gpuDevice,
		std::bind(CNTK::LeakyReLU, _1, L"activation"), L"hidden3");

	//CNTK::Parameter opWeightsParam(CNTK::NDArrayView::RandomUniform<float>({ 1, numHidden_2 }, -0.05, 0.05, 1, gpuDevice));
	//CNTK::Parameter opBiasesParam(CNTK::NDArrayView::RandomUniform<float>({ 1 }, -0.05, 0.05, 1, gpuDevice));
//
	//CNTK::FunctionPtr prediction = CNTK::Tanh(CNTK::Plus(opBiasesParam, CNTK::Times(opWeightsParam, hidden1)), L"output");

	CNTK::FunctionPtr prediction = FullyConnectedDNNLayer(hidden3, 1, gpuDevice,
		std::bind(CNTK::LeakyReLU, _1, L"activation"), L"");

	prediction = CNTK::ElementTimes(prediction, CNTK::Constant::Scalar(2.0f));
	prediction = CNTK::Minus(prediction, CNTK::Constant::Scalar(1.0f), L"output");

	//prediction

	CNTK::Variable target = CNTK::InputVariable({ 1 }, CNTK::DataType::Float);
	CNTK::FunctionPtr trainingLoss = CNTK::SquaredError(prediction, target);
	CNTK::FunctionPtr evaluation = CNTK::ElementDivide(CNTK::Minus(CNTK::Constant::Scalar(2.0f), CNTK::SquaredError(prediction, target)), CNTK::Constant::Scalar(2.0f));

	const size_t minibatchSize = 64;
	const size_t numSweepsToTrainWith = 2000000;
	const size_t numMinibatchesToTrain = numSweepsToTrainWith / minibatchSize;

	CNTK::LearningRateSchedule learningRatePerSample = CNTK::TrainingParameterPerSampleSchedule(0.001);
	CNTK::MomentumSchedule momentumPerSample = CNTK::TrainingParameterPerSampleSchedule(0.001);

	CNTK::Internal::TensorBoardFileWriter tensorboardWriter{ L"tensorboard", prediction };

	CNTK::TrainerPtr trainer = CNTK::CreateTrainer(prediction, trainingLoss, evaluation,
	{ CNTK::SGDLearner(prediction->Parameters(), learningRatePerSample) },
	//{ CNTK::AdamLearner(prediction->Parameters(), learningRatePerSample, momentumPerSample) },
	{} /*TODO: Progress witers*/);

	std::vector<float> inputData(minibatchSize);
	std::vector<float> targetData(minibatchSize);

	std::ranlux48 randEngine(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<float> angleDist(-2 * pi, 2 * pi);

	size_t outputFrequencyInMinibatches = 200;

	for (size_t batchI = 0; batchI < numMinibatchesToTrain; ++batchI)
	{
		for (ptrdiff_t dataI = 0; dataI < minibatchSize; dataI++)
		{
			inputData[dataI] = angleDist(randEngine);
			targetData[dataI] = ::sinf(inputData[dataI]);
		}

		CNTK::ValuePtr batchInput = CNTK::Value::CreateBatch({ 1 }, inputData, gpuDevice, true);
		CNTK::ValuePtr batchTarget = CNTK::Value::CreateBatch({ 1 }, targetData, gpuDevice, true);

		trainer->TrainMinibatch({ { input, batchInput },{ target, batchTarget } }, gpuDevice);


		tensorboardWriter.WriteValue(L"minibatch/avg_loss", trainer->PreviousMinibatchLossAverage(), batchI);
		tensorboardWriter.WriteValue(L"minibatch/avg_metric", trainer->PreviousMinibatchEvaluationAverage(), batchI);

		PrintTrainingProgress(trainer, batchI, outputFrequencyInMinibatches);
	}

	//trainer->SummarizeTrainingProgress();

	tensorboardWriter.Flush();

	tensorboardWriter.Close();
	
	std::vector<float> testAngles = { -pi / 4, 0.0f, pi / 6, pi / 4, pi / 3, pi / 2 };
	CNTK::ValuePtr testInput = CNTK::Value::CreateBatch({ 1 }, testAngles, gpuDevice, true);

	std::vector<std::vector<float>> sines(testAngles.size());
	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = { { prediction, nullptr } };

	prediction->Evaluate({ { input, testInput } }, outputs, gpuDevice);
	outputs[prediction]->CopyVariableValueTo(prediction, sines);

	for (int i = 0; i < testAngles.size(); i++)
	{
		printf("sin(%.0f deg)\t=\t%f\n", testAngles[i] * 180 / pi, sines[i][0]);
	}

	testAngles.resize(721);
	sines.resize(testAngles.size());
	outputs[prediction] = nullptr;

	for (int angle = -360; angle <= 360; angle++)
	{
		testAngles[angle + 360] = angle * pi / 180;
	}

	testInput = CNTK::Value::CreateBatch({ 1 }, testAngles, gpuDevice, true);
	prediction->Evaluate({ { input, testInput } }, outputs, gpuDevice);
	outputs[prediction]->CopyVariableValueTo(prediction, sines);

	std::ofstream fout("sine.csv");

	if (fout.is_open())
	{
		for (int angle = -360; angle <= 360; angle++)
		{
			fout << angle << ", " << sines[angle + 360][0] << "\n";
		}

		fout.close();
	}
	else
	{
		std::cout << "Unable to open: sine.csv" << std::endl;
	}

	if (IsDebuggerPresent())
		system("pause");

    return 0;
}

