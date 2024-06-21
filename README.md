# AMR-attack
### For CNN&LSTM
* Before：<br>
![CM_Before](https://github.com/young-liziii/AMR-attack/blob/main/img/CNN%26LSTM_CM_plot(2).png)
---
* After：<br>
>FGSM
```
# eps = 1/512
_x=eps*torch.sign(x.grad.data)
```
![CM_FGSM](https://github.com/young-liziii/AMR-attack/blob/main/img/CNN%26LSTM_CM_plot(2).png)

>PGD
```
# eps = 1/128, step = eps / float(64), k=15
# 步长取值参考了 Madry 等人，他们提出了在MNIST上 step_size = 2/255 * eps 
for i in range(k):
	adv_x.requires_grad = True
	_y = net(adv_x)
	loss = cross_entropy(_y, y)
	loss.backward()
	adv_x = adv_x + eps * torch.sign(adv_x.grad.data)
```
