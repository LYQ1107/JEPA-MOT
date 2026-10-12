"""Reuse frozen image-pair relations; all trajectory ROI/head decisions remain native."""
import torch
from core.corr import CorrBlock


class SharedMatcher(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.native=model;self.args=model.args
        self.key=None;self.relations=None;self.image_pair_forwards=0;self.association_forwards=0

    def begin_frame(self):
        self.key=None

    def forward(self,image1,image2,detection1,detection2):
        if self.training or torch.is_grad_enabled():
            raise ValueError('Shared matcher is only for frozen inference')
        key=(image1.data_ptr(),image2.data_ptr(),tuple(image1.shape))
        if key!=self.key:
            first=(2*(image1/255.)-1.).contiguous()
            second=(2*(image2/255.)-1.).contiguous()
            with torch.cuda.amp.autocast(enabled=self.args.mixed_precision):
                fmap1,fmap2=self.native.fnet([first,second])
            correlation=CorrBlock(fmap1.float(),fmap2.float(),num_levels=self.args.corr_levels,
                                  radius=self.args.corr_radius)
            corr=correlation(self.native.init(first).detach())
            with torch.cuda.amp.autocast(enabled=self.args.mixed_precision):
                self.relations=self.native.predict_block_similarity.encoder_similarity(corr)
            self.key=key;self.image_pair_forwards+=1
        discrete=detection1.clone().int().detach()
        discrete[:,:,2:4]=torch.max(torch.div(discrete[:,:,2:4],8),torch.tensor([1]).cuda())
        discrete[:,:,4:6]=torch.max(torch.div(discrete[:,:,4:6],8),torch.tensor([1]).cuda())
        with torch.cuda.amp.autocast(enabled=self.args.mixed_precision):
            output=self.native.predict_block_similarity.from_relation_features(
                discrete,self.relations,detection1,detection2)
        self.association_forwards+=1
        return output
