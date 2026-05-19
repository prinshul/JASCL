2 PM - 02/05/25


class StochasticClassifier(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        torch.manual_seed(1024)
        torch.cuda.manual_seed(1024)
        self.mu = nn.Linear(num_features, num_classes,bias=False)
        self.temp = 5
        print('Linear Stochastic Classifier')
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.modules.Linear):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            
    
    def forward(self, x, stochastic=True):
        mu = self.mu.weight
        
        if stochastic:
            weight = torch.randn_like(mu) + mu
        else:
            weight = mu
        
        
        weight = F.normalize(weight, p=2, dim=0)
        x = F.normalize(x, p=2, dim=0)

        score = F.linear(x, weight)
        score = score*self.temp

        return score
		
	
0.5*L_proto

nohup python main.py fit -c configs/step1.yaml --root results/step1 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step0/lightning_logs/version_0/checkpoints/epoch=4-step=1336.ckpt --model.freeze_encoder True > sam_step1_stoch_new.log &



Full stochastic classifier 
temp = 25

nohup python main.py fit -c configs/step1.yaml --root results/step1 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step0/lightning_logs/version_0/checkpoints/epoch=4-step=1336.ckpt --model.freeze_encoder True > sam_step1_stoch_25.log &


Full stochastic classifier
self.temp = 4
+ 0.1*L_proto
nohup python main.py fit -c configs/step1.yaml --root results/step1 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step0/lightning_logs/version_0/checkpoints/epoch=4-step=1336.ckpt --model.freeze_encoder True > sam_step1_stoch_5.log &
