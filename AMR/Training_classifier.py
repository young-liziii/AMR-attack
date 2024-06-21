from rfml.data import build_dataset
from rfml.nn.eval import (compute_accuracy,compute_accuracy_on_cross_sections,compute_confusion,)
from rfml.nn.model import build_model
from rfml.nn.train import build_trainer, PrintingTrainingListener

train, val, test, le = build_dataset(dataset_name="RML2016.10a",path="D:\RML2016.10a_dict.pkl")
# 每个调制信号的size是（2，128），2对应这IQ两路，128对应128个采样点
model = build_model(model_name="alternative_model", input_samples=128, n_classes=len(le))
trainer = build_trainer(strategy="standard", max_epochs=3, gpu=False)  # cpu/gpu
trainer.register_listener(PrintingTrainingListener())
trainer(model=model, training=train, validation=val, le=le)
acc = compute_accuracy(model=model, data=test, le=le)
acc_vs_snr, snr = compute_accuracy_on_cross_sections(model=model, data=test, le=le, column="SNR")
cmn = compute_confusion(model=model, data=test, le=le)


print("===============================")
print("Overall Testing Accuracy: {:.4f}".format(acc))
print("SNR (dB)\tAccuracy (%)")
print("===============================")
for acc, snr in zip(acc_vs_snr, snr):
    print("{snr:d}\t{acc:0.1f}".format(snr=snr, acc=acc * 100))
print("===============================")
print("Confusion Matrix:")
print(cmn)

model.save("CNN+LSTM.pt")