def mist_collate_fn(batch):
    """自定义collate函数，处理MISTDataset返回的额外字段"""
    import sys

    # 调试信息 - 直接写到stderr避免被重定向
    sys.stderr.write(f"[DEBUG] batch type: {type(batch)}\n")
    sys.stderr.write(f"[DEBUG] batch length: {len(batch)}\n")
    if len(batch) > 0:
        sys.stderr.write(f"[DEBUG] batch[0] type: {type(batch[0])}\n")
        if hasattr(batch[0], '__len__'):
            sys.stderr.write(f"[DEBUG] batch[0] length: {len(batch[0])}\n")
            # 打印batch[0]每个元素的类型
            for i, elem in enumerate(batch[0]):
                sys.stderr.write(f"[DEBUG] batch[0][{i}] type: {type(elem)}\n")

    # 处理每个元素（统一处理，不分batch大小）
    try:
        batch_frames = [item[0] for item in batch]
        sys.stderr.write(f"[DEBUG] batch_frames[0] type: {type(batch_frames[0])}\n")
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to extract batch_frames: {e}\n")
        raise

    batch_masks = [item[1] for item in batch]
    batch_img_h = [item[2] for item in batch]
    batch_img_w = [item[3] for item in batch]
    batch_names = [item[4] for item in batch]
    batch_frame_names = [item[5] for item in batch]

    # 堆叠tensor
    frames = torch.stack(batch_frames, dim=0)
    masks = torch.stack(batch_masks, dim=0)
    img_h = torch.tensor(batch_img_h)
    img_w = torch.tensor(batch_img_w)

    return frames, masks, img_h, img_w, batch_names, batch_frame_names
