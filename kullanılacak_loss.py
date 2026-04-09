import torch
import torch.nn as nn

class SensorPhysicsModule(nn.Module):
    def __init__(self, m, k0, alpha, eta, c):
        """
        Sensörün fiziksel sabitleri (Malzeme bilimi parametreleri).
        Bu değerler ya literatürden alınır ya da modelin bir parçası olarak öğrenilir.
        """
        super(SensorPhysicsModule, self).__init__()
        # PyTorch Parameter olarak tanımlıyoruz, istersek eğitebiliriz
        self.m = nn.Parameter(torch.tensor([m], dtype=torch.float32), requires_grad=False)
        self.k0 = nn.Parameter(torch.tensor([k0], dtype=torch.float32), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float32), requires_grad=False)
        self.eta = nn.Parameter(torch.tensor([eta], dtype=torch.float32), requires_grad=False)
        self.c = nn.Parameter(torch.tensor([c], dtype=torch.float32), requires_grad=False)

    def forward(self, a_dirty, states):
        """
        Girdi: Kirli ivme (X,Y,Z) ve YZ'nin tahmin ettiği fiziksel durumlar.
        states boyutu: (Batch, Sequence, 4) -> (D, DT, stress, creep)
        """
        D, DT, sigma, creep = states[:,:,0], states[:,:,1], states[:,:,2], states[:,:,3]
        
        # 1. DENKLEM: k_act hesaplama (Görseldeki k_hat denklemi)
        # k_act = k0 * (1-D) * (1 + alpha*DT) * (1 - eta*sigma)
        k_act_scale = (1.0 - D) * (1.0 + self.alpha * DT) * (1.0 - self.eta * sigma)
        k_act = self.k0 * k_act_scale
        
        # Sönümleme (c*dot_x) terimini MEMS sensörler için genellikle ihmal edebiliriz, 
        # çünkü bu dinamik kalibrasyonla çözülür. Biz Bias ve Esneklik değişimine odaklanacağız.

        # 2. DENKLEM: a_clean (ddot_x) hesaplama. 
        # m*ddot_x = F_dirty - k_act * (x_internal - creep)
        # MEMS'te ivme, yayın iç yer değiştirmesiyle orantılıdır (x_internal \approx m*a_dirty / k0).
        # Buradan türetilen düzeltme formulü (Hatalı ölçeklemeyi ve creep'i düzeltmek):
        
        # İstenmeyen sabit sapmayı (Creep/Bias) çıkar
        a_unbiased = a_dirty - creep
        
        # Degradasyona uğramış yay katsayısına (k_act) göre ivmeyi yeniden ölçekle (correct scale)
        # k_act_scale ideal durumda 1'dir. Eğer sensör bozulmuşsa bu değer düşer, 
        # biz ivmeyi düzelterek artırırız.
        a_clean = a_unbiased / (k_act_scale + 1e-6) # Sıfıra bölünmeyi engellemek için epsilon
        
        return a_clean

class PhysicsInformedKinematicLoss(nn.Module):
    def __init__(self, dt, m, k0, alpha, eta, c):
        super(PhysicsInformedKinematicLoss, self).__init__()
        self.dt = dt
        # Fizik motorumuzu buraya entegre ediyoruz
        self.physics_engine = SensorPhysicsModule(m, k0, alpha, eta, c)
        self.mse_loss = nn.MSELoss()

    def forward(self, a_dirty, states, p_gt, v_0, p_0):
        """
        Girdiler: Kirli ivme, YZ'nin fizik parametre tahminleri, Gerçek konum (GT), Başlangıç hız/konum
        """
        
        # 1. ADIM: Kirli veriyi FİZİK DENKLEMİNDEN geçirip düzelt
        # YZ parametreleri tahmin etti, denklem bunları kullanarak ivmeyi düzeltti.
        a_clean_pred = self.physics_engine(a_dirty, states)
        
        # 2. ADIM: KİNEMATİK ADIM (Bir önceki anlaştığımız integral)
        # Düzeltilmiş ivmeyi konuma çeviriyoruz.
        # İvmeden hıza integral
        v_delta = torch.cumsum(a_clean_pred * self.dt, dim=1)
        v_pred = v_0.unsqueeze(1) + v_delta
        
        # Hızdan konuma integral
        p_delta = torch.cumsum(v_pred * self.dt, dim=1)
        p_pred = p_0.unsqueeze(1) + p_delta
        
        # 3. ADIM: KAYIP HESAPLAMA (Gözetimli Öğrenme)
        # Hesaplanan konum ile veri setindeki GERÇEK konumu karşılaştırıyoruz.
        # YZ, bu MSE'yi düşürmek için fizik parametrelerini (D, DT vb.) nasıl tahmin etmesi gerektiğini öğrenecek.
        loss_position = self.mse_loss(p_pred, p_gt)
        
        # Opsiyonel: Fiziksel parametrelere kısıtlamalar ekleyebiliriz (Regülarizasyon)
        # Örn: Hasar (D) asla negatif olamaz.
        loss_damage_constraint = torch.mean(torch.relu(-states[:,:,0])) # D < 0 ise cezalandır
        
        total_loss = loss_position + 0.1 * loss_damage_constraint
        
        return total_loss