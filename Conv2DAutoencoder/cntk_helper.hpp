#pragma once

#ifdef __INTELLISENSE__
#include <CNTKLibrary.h>
#endif


inline CNTK::Parameter NewWeightParameter(const CNTK::NDShape& shape, const CNTK::DeviceDescriptor& device, const std::wstring& outputName = L"")
{
	return CNTK::Parameter(shape, CNTK::DataType::Float, 
		CNTK::GlorotUniformInitializer(1, CNTK::SentinelValueForInferParamInitRank, CNTK::SentinelValueForInferParamInitRank,
		//CNTK::NormalInitializer(0.05, CNTK::SentinelValueForInferParamInitRank, CNTK::SentinelValueForInferParamInitRank,
		std::chrono::high_resolution_clock::now().time_since_epoch().count()), device, outputName);
}

inline CNTK::Parameter NewBiasParameter(const CNTK::NDShape& shape, const CNTK::DeviceDescriptor& device, const std::wstring& outputName = L"")
{
	return CNTK::Parameter(shape, CNTK::DataType::Float, 0.0f, device, outputName);
}

inline CNTK::FunctionPtr FullyConnectedLinearLayer(CNTK::Variable input, size_t outputDim,
	const CNTK::DeviceDescriptor& device, const std::wstring& outputName = L"")
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	CNTK::Parameter timesParam({ outputDim, inputDim }, CNTK::DataType::Float,
		CNTK::GlorotUniformInitializer(1, CNTK::SentinelValueForInferParamInitRank, CNTK::SentinelValueForInferParamInitRank,
			//CNTK::NormalInitializer(0.05, CNTK::SentinelValueForInferParamInitRank, CNTK::SentinelValueForInferParamInitRank,
			std::chrono::high_resolution_clock::now().time_since_epoch().count()), device, outputName + L"/weights");
	CNTK::FunctionPtr timesFunction = CNTK::Times(timesParam, input, outputName + L"/matmul");

	CNTK::Parameter plusParam({ outputDim }, 0.0f, device, outputName + L"/biases");
	return CNTK::Plus(plusParam, timesFunction, outputName + L"/output");
}


inline CNTK::FunctionPtr FullyConnectedLinearLayer_SharedWeight(CNTK::Variable input, 
	const CNTK::Variable& weightParam, const CNTK::Variable& biasParam, const std::wstring& outputName = L"")
{
	assert(input.Shape().Rank() == 1);

	CNTK::FunctionPtr timesFunction = CNTK::Times(weightParam, input, outputName + L"/matmul");

	return CNTK::Plus(biasParam, timesFunction, outputName + L"/output");
}


inline CNTK::FunctionPtr FullyConnectedLinearLayer_UniformInit(CNTK::Variable input, size_t outputDim,
	double scale,
	const CNTK::DeviceDescriptor& device, const std::wstring& outputName = L"")
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	CNTK::Parameter timesParam({ outputDim, inputDim }, CNTK::DataType::Float,
		CNTK::UniformInitializer(scale, std::chrono::high_resolution_clock::now().time_since_epoch().count()), device, outputName + L"/weights");
	CNTK::FunctionPtr timesFunction = CNTK::Times(timesParam, input, outputName + L"/matmul");

	CNTK::Parameter plusParam({ outputDim }, 0.0f, device, outputName + L"/biases");
	return CNTK::Plus(plusParam, timesFunction, outputName + L"/output");
}

inline CNTK::FunctionPtr FullyConnectedDNNLayer(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device,
	const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&, std::wstring)>& nonLinearity, const std::wstring& outputName = L"",
	const std::wstring& activationName = L"activation")
{
	return nonLinearity(FullyConnectedLinearLayer(input, outputDim, device, outputName), outputName + L"/" + activationName);
}

inline CNTK::FunctionPtr FullyConnectedDNNLayer_SharedWeight(CNTK::Variable input, const CNTK::Variable& weightParam, const CNTK::Variable& biasParam,
	const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&, std::wstring)>& nonLinearity, const std::wstring& outputName = L"",
	const std::wstring& activationName = L"activation")
{
	return nonLinearity(FullyConnectedLinearLayer_SharedWeight(input, weightParam, biasParam, outputName), outputName + L"/" + activationName);
}

inline CNTK::FunctionPtr BatchNormalizationLayer(CNTK::Variable input, const CNTK::DeviceDescriptor& device,
	const std::wstring& outputName = L"", bool inputIsConvOutput = false)
{
	auto biasParams = CNTK::Parameter({ CNTK::NDShape::InferredDimension }, 0.0f, device, outputName + L"/bias");
	auto scaleParams = CNTK::Parameter({ CNTK::NDShape::InferredDimension }, 1.0f, device, outputName + L"/scale");
	auto runningMean = CNTK::Constant({ CNTK::NDShape::InferredDimension }, 0.0f, device, outputName + L"/running_mean");
	auto runningInvStd = CNTK::Constant({ CNTK::NDShape::InferredDimension }, 0.0f, device, outputName + L"/running_inv_std");
	auto runningCount = CNTK::Constant({}, 0.0f, device, outputName + L"/running_count");
	return BatchNormalization(input, scaleParams, biasParams, runningMean, runningInvStd, runningCount, inputIsConvOutput, 5000.0, 0.0, 1e-5 /* epsilon */,
		true, false, outputName + L"/bn_op");
}


inline CNTK::FunctionPtr Conv2DLayer(const CNTK::DeviceDescriptor& device, CNTK::Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, 
	bool autoPadding = true, size_t hStride = 1, size_t vStride = 1,
	double glorotScale = 1.0, const std::wstring& outputName = L"")
{
	size_t numInputChannels = input.Shape()[input.Shape().Rank() - 1];

	auto convParams = CNTK::Parameter({ kernelWidth, kernelHeight, numInputChannels, outFeatureMapCount }, 
		CNTK::DataType::Float, CNTK::GlorotUniformInitializer(glorotScale, -1, 2), device);
	auto convFunction = CNTK::Convolution(convParams, input, { hStride, vStride, numInputChannels }, { true }, { autoPadding });
	convFunction->SetName(outputName + L"/conv2d");

	return convFunction;
}

inline CNTK::FunctionPtr ConvTranspose2DLayer(const CNTK::DeviceDescriptor& device, CNTK::Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight,
	bool autoPadding = true, size_t hStride = 1, size_t vStride = 1,
	double glorotScale = 1.0, const std::wstring& outputName = L"")
{
	size_t numInputChannels = input.Shape()[input.Shape().Rank() - 1];

	auto convParams = CNTK::Parameter({ kernelWidth, kernelHeight, numInputChannels, outFeatureMapCount },
		CNTK::DataType::Float, CNTK::GlorotUniformInitializer(glorotScale, -1, 2), device);
	auto convFunction = CNTK::ConvolutionTranspose(convParams, input, { hStride, vStride, numInputChannels }, { true }, { autoPadding });
	convFunction->SetName(outputName + L"/conv_transpose2d");

	return convFunction;
}


inline CNTK::FunctionPtr HuberLoss(CNTK::Variable y_true, CNTK::Variable y_pred, float delta = 1.0f)
{
	CNTK::FunctionPtr err = CNTK::Minus(y_true, y_pred);

	CNTK::FunctionPtr cond = CNTK::Less(CNTK::Abs(err), CNTK::Constant::Scalar(delta));
	CNTK::FunctionPtr L2 = CNTK::ElementTimes(CNTK::Constant::Scalar(0.5f), CNTK::Square(err));
	CNTK::FunctionPtr L1 = CNTK::Abs(err);
	L1 = CNTK::Minus(L1, CNTK::Constant::Scalar(0.5f * delta));
	L1 = CNTK::ElementTimes(CNTK::Constant::Scalar(delta), L1);

	return CNTK::ElementSelect(cond, L2, L1);
}

template <typename ElementType>
inline CNTK::FunctionPtr Stabilize(const CNTK::Variable& x, const CNTK::DeviceDescriptor& device)
{
	ElementType scalarConstant = 4.0f;
	auto f = CNTK::Constant::Scalar(scalarConstant);
	auto fInv = CNTK::Constant::Scalar(f.GetDataType(), 1.0 / scalarConstant);

	auto beta = CNTK::ElementTimes(fInv, CNTK::Log(CNTK::Constant::Scalar(f.GetDataType(), 1.0) +
		CNTK::Exp(CNTK::ElementTimes(f, CNTK::Parameter({}, f.GetDataType(), 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
	return CNTK::ElementTimes(beta, x);
}

template <typename ElementType>
std::pair<CNTK::FunctionPtr, CNTK::FunctionPtr> LSTMPCellWithSelfStabilization(CNTK::Variable input, CNTK::Variable prevOutput, CNTK::Variable prevCellState,
	const CNTK::DeviceDescriptor& device)
{
	size_t outputDim = prevOutput.Shape()[0];
	size_t cellDim = prevCellState.Shape()[0];

	auto createBiasParam = [device](size_t dim) {
		return CNTK::Parameter({ dim }, (ElementType)0.0, device);
	};

	unsigned long seed2 = 1;
	auto createProjectionParam = [device, &seed2](size_t outputDim) {
		return CNTK::Parameter({ outputDim, CNTK::NDShape::InferredDimension }, CNTK::AsDataType<ElementType>(), CNTK::GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
	};

	auto createDiagWeightParam = [device, &seed2](size_t dim) {
		return CNTK::Parameter({ dim }, CNTK::AsDataType<ElementType>(), CNTK::GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
	};

	auto stabilizedPrevOutput = Stabilize<ElementType>(prevOutput, device);
	auto stabilizedPrevCellState = Stabilize<ElementType>(prevCellState, device);

	auto projectInput = [input, cellDim, createBiasParam, createProjectionParam]() {
		return createBiasParam(cellDim) + CNTK::Times(createProjectionParam(cellDim), input);
	};

	// Input gate
	auto it = CNTK::Sigmoid(projectInput() + CNTK::Times(createProjectionParam(cellDim), stabilizedPrevOutput) +
		CNTK::ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
	auto bit = CNTK::ElementTimes(it, CNTK::Tanh(projectInput() + CNTK::Times(createProjectionParam(cellDim), stabilizedPrevOutput)));

	// Forget-me-not gate
	auto ft = CNTK::Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) +
		CNTK::ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
	auto bft = CNTK::ElementTimes(ft, prevCellState);

	auto ct = bft + bit;

	// Output gate
	auto ot = CNTK::Sigmoid(projectInput() + CNTK::Times(createProjectionParam(cellDim), stabilizedPrevOutput) + CNTK::ElementTimes(createDiagWeightParam(cellDim),
		Stabilize<ElementType>(ct, device)));
	auto ht = CNTK::ElementTimes(ot, CNTK::Tanh(ct));

	auto c = ct;
	auto h = (outputDim != cellDim) ? CNTK::Times(createProjectionParam(outputDim), Stabilize<ElementType>(ht, device)) : ht;

	return{ h, c };
}

template <typename ElementType>
std::pair<CNTK::FunctionPtr, CNTK::FunctionPtr> LSTMPComponentWithSelfStabilization(CNTK::Variable input,
	const CNTK::NDShape& outputShape,
	const CNTK::NDShape& cellShape,
	const std::function<CNTK::FunctionPtr(const CNTK::Variable&)>& recurrenceHookH,
	const std::function<CNTK::FunctionPtr(const CNTK::Variable&)>& recurrenceHookC,
	const CNTK::DeviceDescriptor& device)
{
	auto dh = PlaceholderVariable(outputShape, input.DynamicAxes());
	auto dc = PlaceholderVariable(cellShape, input.DynamicAxes());

	auto LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc, device);

	auto actualDh = recurrenceHookH(LSTMCell.first);
	auto actualDc = recurrenceHookC(LSTMCell.second);

	// Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
	LSTMCell.first->ReplacePlaceholders({ { dh, actualDh },{ dc, actualDc } });

	return{ LSTMCell.first, LSTMCell.second };
}

inline CNTK::FunctionPtr Embedding(const CNTK::Variable& input, size_t embeddingDim, const CNTK::DeviceDescriptor& device)
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];
	auto embeddingParameters = CNTK::Parameter({ embeddingDim, inputDim }, CNTK::DataType::Float, CNTK::GlorotUniformInitializer(), device);
	return Times(embeddingParameters, input);
}

inline CNTK::FunctionPtr LSTMSequenceClassifierNet(const CNTK::Variable& input, size_t numOutputClasses, size_t embeddingDim, size_t LSTMDim, size_t cellDim,
	const CNTK::DeviceDescriptor& device, const std::wstring& outputName = L"")
{
	auto embeddingFunction = Embedding(input, embeddingDim, device);
	auto pastValueRecurrenceHook = [](const CNTK::Variable& x) { return CNTK::PastValue(x); };
	auto LSTMFunction = LSTMPComponentWithSelfStabilization<float>(embeddingFunction, { LSTMDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;
	auto thoughtVectorFunction = CNTK::Sequence::Last(LSTMFunction);

	return FullyConnectedLinearLayer(thoughtVectorFunction, numOutputClasses, device, outputName);
}