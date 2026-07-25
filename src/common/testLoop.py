import torch

def testModel(testData, model, evaluator, device):
    """
        Runs model evaluation on the test dataset.

        The test loop is shared across all NLP tasks. Task-specific metric calculations are handled by the evaluator,
        allowing different output formats and evaluation strategies without changing the evaluation flow.
    """
    model.eval()
    # Disable gradient tracking as no parameter update in testing.
    with torch.inference_mode():
        for batch in testData:
            x = batch["input_ids"].to(device)
            attentionMask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            # Mixed precision reduces memory usage while still converting logits back to float32 for numerical stability in metric calculations
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids = x, attention_mask = attentionMask, labels = y).logits.float()

            # Standardise output shape for evaluator input.
            # NER outputs [batch, sequence, classes] and is flattened into
            # individual token predictions. Other tasks remain unchanged.
            output = output.view(-1, output.size(-1))
            y = y.view(-1)

            evaluator.step(output, y)

        # Evaluators accumulate predictions across batches before calculating the final result.
        result = evaluator.fullEval()

    
    return result

