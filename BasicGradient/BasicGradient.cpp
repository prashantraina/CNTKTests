// BasicGradient.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int wmain()
{
	CNTK::DeviceDescriptor device = CNTK::DeviceDescriptor::GPUDevice(0);

	//CNTK::Variable x = CNTK::PlaceholderVariable(CNTK::NDShape::Unknown(), CNTK::DataType::Float, L"x");
	//CNTK::Variable y = CNTK::PlaceholderVariable(CNTK::NDShape::Unknown(), CNTK::DataType::Float, L"y");

	CNTK::Variable x = CNTK::InputVariable({}, CNTK::DataType::Float, true, L"x");
	CNTK::Variable y = CNTK::InputVariable({}, CNTK::DataType::Float, true, L"y");

	CNTK::FunctionPtr z = CNTK::Plus(CNTK::Square(x), CNTK::Pow(y, CNTK::Constant::Scalar(3.0f)));
	CNTK::FunctionPtr z_mean = CNTK::ReduceMean(z, CNTK::Axis::AllAxes());
	CNTK::FunctionPtr z_neg = CNTK::Negate(z);

	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = { { z, nullptr} };
	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> gradients = { { x, nullptr }, { y, nullptr } };

	std::vector<float> xValues = { 5.0f, 7.0f, 8.0f };
	std::vector<float> yValues = { 4.0f, 4.0f, 4.0f };

	CNTK::ValuePtr xValuePtr = CNTK::Value::CreateBatch({}, xValues , device, true);
	CNTK::ValuePtr yValuePtr = CNTK::Value::CreateBatch({}, yValues, device, true);

	//std::wcout << xValuePtr->Shape().AsString() << std::endl;;

	//CNTK::FunctionPtr z_tmp = z->ReplacePlaceholders({ { x, CNTK::Constant::Scalar(5.0f) },{ y, CNTK::Constant::Scalar(4.0f) } });

	z->Evaluate({ { x, xValuePtr }, {y, yValuePtr} }, outputs, device);
	//z_tmp->Evaluate({}, outputs, device);

	std::vector<std::vector<float>> output_floats;
	outputs[z]->CopyVariableValueTo(z, output_floats);

	//z_tmp->Gradients({}, gradients, device);
	z_mean->Gradients({ { x, xValuePtr },{ y, yValuePtr } }, gradients, device);


	std::vector<std::vector<float>> gradients_x;
	std::vector<std::vector<float>> gradients_y;

	gradients[x]->CopyVariableValueTo(x, gradients_x);
	gradients[y]->CopyVariableValueTo(x, gradients_y);


	if (IsDebuggerPresent())
		system("pause");

    return 0;
}

