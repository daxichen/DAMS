import numpy as np
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from .vit_pytorch import VisionTransformer
from collections import OrderedDict
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from supcontrast import SupConLoss
_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, dtype, token_embedding):
        super().__init__()
        dtype = dtype
        ctx_dim = 512
        ctx_init = "A hyperspectral image of X."
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 1
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        token_embedding = token_embedding.cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        
        n_cls_ctx = 1
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :]) 

        self.n_ctx = n_ctx

    def forward(self, cls_ctx):
        cls_ctx = cls_ctx.unsqueeze(1)
        b = cls_ctx.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts

def l2_norm(input, axis=1):
  norm = torch.norm(input, 2, axis, True)
  output = torch.div(input, norm)
  return output

# feature embedders
class feature_embedder(nn.Module):

  def __init__(self):
    super(feature_embedder, self).__init__()
    self.bottleneck_layer_fc = nn.Linear(768, 512)
    self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
    self.bottleneck_layer_fc.bias.data.fill_(0.1)
    self.bottleneck_layer = nn.Sequential(self.bottleneck_layer_fc, nn.ReLU(),
                                          nn.Dropout(0.5))

  def forward(self, input, norm_flag=True):
    feature = self.bottleneck_layer(input)
    if (norm_flag):
      feature_norm = torch.norm(
          feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)**0.5 * (2)**0.5
      feature = torch.div(feature, feature_norm)
    return feature

# classifier
class classifier(nn.Module):

  def __init__(self):
    super(classifier, self).__init__()
    self.classifier_layer = nn.Linear(512, 2)
    self.classifier_layer.weight.data.normal_(0, 0.01)
    self.classifier_layer.bias.data.fill_(0.0)

  def forward(self, input, norm_flag=True):
    if (norm_flag):
      self.classifier_layer.weight.data = l2_norm(
          self.classifier_layer.weight, axis=0)
      classifier_out = self.classifier_layer(input)
    else:
      classifier_out = self.classifier_layer(input)
    return classifier_out

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class ProjectHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, out_dim=256):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, feat):
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    
class EncoderTrans(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(EncoderTrans, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        feat = self.enc_net(feat)
        return feat
    
# HyperCLIP
class HyperCLIP(nn.Module):
    def __init__(self, args: int,
                    embed_dim,
                    clip_model,
                    classnames,
                    dtype,
                # vision
                    img_size,
                    inchannel,
                    layers,
                    vision_patch_size: int,
                    num_classes,
                    # text
                    context_length: int,
                    vocab_size: int,
                    transformer_width: int,
                    transformer_heads: int,
                    transformer_layers: int
                    ):
        super(HyperCLIP, self).__init__()
       
        self.context_length = context_length
        
        self.visual = VisionTransformer(img_size=img_size,
                                        in_c=inchannel,
                                        patch_size=1,
                                        embed_dim=int(embed_dim),
                                        depth=layers,
                                        num_heads=4,
                                        representation_size=None,
                                        num_classes=num_classes
                                        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.mlp_i2t = EncoderTrans(input_dim=embed_dim, hidden=256, out_dim=512)
        self.mlp_t2i = EncoderTrans(input_dim=embed_dim, hidden=256, out_dim=512)
        
        self.i_proj = ProjectHead(input_dim=512, hidden_dim=256, out_dim=256)
        self.t_proj = ProjectHead(input_dim=512, hidden_dim=256, out_dim=256)
        
        self.temperature = args.temperature
        self.criterion = nn.CrossEntropyLoss()
        self.SupConLoss = SupConLoss(temperature=args.temperature)
        self.cosine_similarity = nn.CosineSimilarity()
        self.mse_loss = nn.MSELoss()
        self.device = args.gpu
        self.n_views = 2

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def torch_cosine_similarity(self, features1, features2):
        norm1 = torch.norm(features1, dim=-1).reshape(features1.shape[0], 1)
        norm2 = torch.norm(features2, dim=-1).reshape(1, features2.shape[0])
        end_norm = torch.mm(norm1.to(torch.double), norm2.to(torch.double))
        cos = torch.mm(features1.to(torch.double), features2.T.to(torch.double)) / end_norm
        return cos.mean()
    
    def Reg(self, w1, w2):  # orthogonality regulation
        w1 = F.normalize(w1, dim=-1)  # l2-normalize
        w2 = F.normalize(w2, dim=-1)  # l2-normalize
        reg = torch.matmul(w1, w2.T)
        return torch.mean(torch.sum(reg ** 2, dim=-1))

    @property
    def dtype(self):
        # return self.visual.DR.weight.dtype
        # return self.visual.fc.weight.dtype
        return self.visual.head.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    # def encode_text(self, prompts, tokenized_prompts):
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def info_nce_loss(self, feature_view_1, feature_view_2):
        assert feature_view_1.shape == feature_view_2.shape
        features = torch.cat([feature_view_1, feature_view_2], dim=0)

        labels = torch.cat([torch.arange(feature_view_1.shape[0]) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        
        loss_nce = self.criterion(logits, labels)
        return loss_nce
    
    def forward_stage1(self, image, aug_img, text):
        text_features = self.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features, class_logits = self.encode_image(image)
        v_dim = int(image_features.shape[1] / 2)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        aug_image_features, aug_class_logits = self.encode_image(aug_img)
        aug_image_features = aug_image_features / aug_image_features.norm(dim=-1, keepdim=True)
        return image_features, aug_image_features, text_features
    
    def forward(self, image, aug_img, text, label):
        text_features = self.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features, class_logits = self.encode_image(image)
        v_dim = int(image_features.shape[1] / 2)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        aug_image_features, aug_class_logits = self.encode_image(aug_img)
        aug_image_features = aug_image_features / aug_image_features.norm(dim=-1, keepdim=True)
        loss_ce = self.criterion(class_logits,label) + self.criterion(aug_class_logits, label)
        
        t_emd_t = self.mlp_i2t(image_features)
        t_emd_t2 = self.mlp_i2t(aug_image_features)
        i_emd_t = self.mlp_t2i(text_features)        
        
        t_emd_t = t_emd_t/torch.norm(t_emd_t, dim=1, keepdim=True)
        t_emd_t2 = t_emd_t2/torch.norm(t_emd_t2, dim=1, keepdim=True)
        i_emd_t = i_emd_t/torch.norm(i_emd_t, dim=1, keepdim=True)
        sim_loss = self.info_nce_loss(t_emd_t, t_emd_t2) #+ self.info_nce_loss(i_emd_t, i_emd_t2)
        
        i2t_loss = torch.mean(torch.norm(t_emd_t-text_features/torch.norm(text_features, dim=1, keepdim=True), dim=1))
        i2t2_loss = torch.mean(torch.norm(t_emd_t2-text_features/torch.norm(text_features, dim=1, keepdim=True), dim=1))
        t2i_loss = torch.mean(torch.norm(i_emd_t-image_features/torch.norm(image_features, dim=1, keepdim=True), dim=1))
        
        loss_trans = (i2t_loss + i2t2_loss + t2i_loss)/1
        
        i_emd_proj = self.i_proj(image_features)
        augi_emd_proj = self.i_proj(aug_image_features)
        t_emd_proj = self.t_proj(text_features)
        
        i_t_emd_proj = torch.cat([i_emd_proj.unsqueeze(1), t_emd_proj.unsqueeze(1)], dim=1)
        augi_t_emd_proj = torch.cat([augi_emd_proj.unsqueeze(1), t_emd_proj.unsqueeze(1)], dim=1)
        
        loss_scl = self.SupConLoss(i_t_emd_proj, label) + self.SupConLoss(augi_t_emd_proj, label)
        loss_e = F.mse_loss(image_features[:, :v_dim], image_features[:, v_dim:]) + F.mse_loss(text_features[:, :v_dim], text_features[:, v_dim:]) + F.mse_loss(aug_image_features[:, :v_dim], aug_image_features[:, v_dim:]) 
        
        return sim_loss, loss_scl, loss_ce, loss_trans, loss_e
    
    def forward_eval(self, input):
        image_features, class_logits = self.encode_image(input)
        return class_logits
    
    
