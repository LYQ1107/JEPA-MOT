"""
Convert a DARTH checkpoint for inference:
Keep ALL original keys, but overwrite student weights with teacher weights:
  detector.*    <- teacher_detector.*
  track_head.*  <- teacher_track_head.*
This makes the checkpoint behave like a standard QDTrack model
while carrying the adapted (teacher) parameters.
"""
import argparse, torch, os

def convert(src_path, dst_path):
    ckpt = torch.load(src_path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)

    # Copy full state dict, then overwrite student keys with teacher values
    new_sd = dict(sd)
    overwritten = 0
    for k, v in sd.items():
        if k.startswith('teacher_detector.'):
            student_k = k.replace('teacher_detector.', 'detector.', 1)
            if student_k in new_sd:
                new_sd[student_k] = v
                overwritten += 1
        elif k.startswith('teacher_track_head.'):
            student_k = k.replace('teacher_track_head.', 'track_head.', 1)
            if student_k in new_sd:
                new_sd[student_k] = v
                overwritten += 1

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt['state_dict'] = new_sd
        out = ckpt
    else:
        out = new_sd

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    torch.save(out, dst_path)
    print(f"Saved: {dst_path}  (total={len(new_sd)} keys, overwritten={overwritten})")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('dst')
    args = p.parse_args()
    convert(args.src, args.dst)
