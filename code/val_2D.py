import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
import h5py

def calculate_metric_percase_save(pred, gt, spacing):
    if pred.sum() > 0 and gt.sum() > 0:
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    else:
        dice = 0.0
        hd95 = 100.0
        asd = 20.0
    return dice, hd95, asd


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        #hd95 = metric.binary.hd95(pred, gt)
        hd95 = dice
        return dice, hd95
    else:
        return 0, 0

def test_single_volume_co_save(case, net1, net2, test_save_path, FLAGS, batch_size=12):
    import os
    os.makedirs(test_save_path, exist_ok=True)
    #print (FLAGS.root_path + "/test_volumes/{}".format(case))
    h5f = h5py.File(FLAGS.root_path + "/test_volumes/{}".format(case), 'r')
    
    image = h5f['image'][:]
    label = h5f['label'][:]

    try:
        spacing = h5f['spacing'][:]
    except:
        spacing = [1,1,1]
    #print(spacing)

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        ind_x = np.array([i for i in range(image.shape[0])])
        for ind in ind_x[::batch_size]:
            if ind + batch_size < image.shape[0]:
                slice = image[ind:ind + batch_size, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net1.eval()
                net2.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        0.5*(net1(input)+net2(input)), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:ind + batch_size, ...] = pred
            else:
                slice = image[ind:, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net1.eval()
                net2.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        0.5*(net1(input)+net2(input)), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:, ...] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net1.eval()
        net2.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                0.5*(net1(input)+net2(input)), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    case = case.replace(".h5", "")
    
    metric_list = []
    for i in range(1, FLAGS.num_classes):
        metric_list.append(calculate_metric_percase_save(
            prediction == i, label == i,spacing=(spacing[2], spacing[0], spacing[1])))
        
        
    #print(image.shape)
    #print(label.shape)
    #print(prediction.shape)
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing(spacing)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.int16))
    prd_itk.SetSpacing(spacing)
    lab_itk = sitk.GetImageFromArray(label.astype(np.int16))
    lab_itk.SetSpacing(spacing)
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))
    return np.array(metric_list)

def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out = net(input)
                out = torch.argmax(out, dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_co_save(case, net1, net2, test_save_path, FLAGS, batch_size=12):
    import os
    os.makedirs(test_save_path, exist_ok=True)
    #print (FLAGS.root_path + "/test_volumes/{}".format(case))
    h5f = h5py.File(FLAGS.root_path + "/test_volumes/{}".format(case), 'r')
    
    image = h5f['image'][:]
    label = h5f['label'][:]


    try:
        spacing = h5f['spacing'][:]
    except:
        spacing = [1,1,1]
    #print(spacing)

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        ind_x = np.array([i for i in range(image.shape[0])])
        for ind in ind_x[::batch_size]:
            if ind + batch_size < image.shape[0]:
                slice = image[ind:ind + batch_size, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net1.eval()
                net2.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        0.5*(net1(input)+net2(input)), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:ind + batch_size, ...] = pred
            else:
                slice = image[ind:, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net1.eval()
                net2.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        0.5*(net1(input)+net2(input)), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:, ...] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net1.eval()
        net2.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                0.5*(net1(input)+net2(input)), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    case = case.replace(".h5", "")
    
    metric_list = []
    for i in range(1, FLAGS.num_classes):
        metric_list.append(calculate_metric_percase_save(
            prediction == i, label == i,spacing=(spacing[2], spacing[0], spacing[1])))
        
        
    #print(image.shape)
    #print(label.shape)
    #print(prediction.shape)
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing(spacing)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.int16))
    prd_itk.SetSpacing(spacing)
    lab_itk = sitk.GetImageFromArray(label.astype(np.int16))
    lab_itk.SetSpacing(spacing)
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))
    return np.array(metric_list)


def test_single_volume_val(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out = net(input)
                out = torch.argmax(out, dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list,  prediction


def test_co_volume_val(image, label, net1, net2, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net1.eval()
            net2.eval()
            with torch.no_grad():
                out = (net1(input)+net2(input))*0.5
                out = torch.argmax(out, dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net1.eval()
        net2.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                0.5*(net1(input)+net2(input)), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list,  prediction



def test_single_volume_co_save(case, net1, net2, test_save_path, FLAGS, batch_size=12,patch_size = [256,256]):
    import os
    os.makedirs(test_save_path, exist_ok=True)
    #print (FLAGS.root_path + "/test_volumes/{}".format(case))
    h5f = h5py.File(FLAGS.root_path + "/test_volumes/{}".format(case), 'r')
    
    image = h5f['image'][:]
    label = h5f['label'][:]
    try:
        spacing = h5f['spacing'][:]
    except:
        spacing = [1,1,1]
    #print(spacing)

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        ind_x = np.array([i for i in range(image.shape[0])])
        for ind in ind_x[::batch_size]:
            if ind + batch_size < image.shape[0]:
                slice = image[ind:ind + batch_size, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net1.eval()
                net2.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        0.5*(net1(input)+net2(input)), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:ind + batch_size, ...] = pred
            else:
                slice = image[ind:, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net1.eval()
                net2.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        0.5*(net1(input)+net2(input)), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:, ...] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net1.eval()
        net2.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                0.5*(net1(input)+net2(input)), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    case = case.replace(".h5", "")
    
    metric_list = []
    for i in range(1, FLAGS.num_classes):
        metric_list.append(calculate_metric_percase_save(
            prediction == i, label == i,spacing=(spacing[2], spacing[0], spacing[1])))
        
        
    #print(image.shape)
    #print(label.shape)
    #print(prediction.shape)
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing(spacing)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.int16))
    prd_itk.SetSpacing(spacing)
    lab_itk = sitk.GetImageFromArray(label.astype(np.int16))
    lab_itk.SetSpacing(spacing)
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))
    return np.array(metric_list)


def test_single_volume_single_save(case, net, test_save_path, FLAGS, batch_size=12):
    import os
    os.makedirs(test_save_path, exist_ok=True)
    #print (FLAGS.root_path + "/test_volumes/{}".format(case))
    h5f = h5py.File(FLAGS.root_path + "/test_volumes/{}".format(case), 'r')

    
    image = h5f['image'][:]
    label = h5f['label'][:]


    try:
        spacing = h5f['spacing'][:]
    except:
        spacing = [1,1,1]
    #print(spacing)

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        ind_x = np.array([i for i in range(image.shape[0])])
        for ind in ind_x[::batch_size]:
            if ind + batch_size < image.shape[0]:
                slice = image[ind:ind + batch_size, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net.eval()

                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        net(input), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:ind + batch_size, ...] = pred
            else:
                slice = image[ind:, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net.eval()
                
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        net(input), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:, ...] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()

        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    case = case.replace(".h5", "")
    
    metric_list = []
    for i in range(1, FLAGS.num_classes):
        metric_list.append(calculate_metric_percase_save(
            prediction == i, label == i,spacing=(spacing[2], spacing[0], spacing[1])))
        
        
    #print(image.shape)
    #print(label.shape)
    #print(prediction.shape)
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing(spacing)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.int16))
    prd_itk.SetSpacing(spacing)
    lab_itk = sitk.GetImageFromArray(label.astype(np.int16))
    lab_itk.SetSpacing(spacing)
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))
    return np.array(metric_list)




def test_single_volume_single_save_cct(case, net, test_save_path, FLAGS, batch_size=12):
    import os
    os.makedirs(test_save_path, exist_ok=True)
    #print (FLAGS.root_path + "/test_volumes/{}".format(case))
    h5f = h5py.File(FLAGS.root_path + "/test_volumes/{}".format(case), 'r')

    
    image = h5f['image'][:]
    label = h5f['label'][:]


    try:
        spacing = h5f['spacing'][:]
    except:
        spacing = [1,1,1]
    #print(spacing)

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        ind_x = np.array([i for i in range(image.shape[0])])
        for ind in ind_x[::batch_size]:
            if ind + batch_size < image.shape[0]:
                slice = image[ind:ind + batch_size, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net.eval()

                with torch.no_grad():
                    
                    out1,out2  = net(input,mutli = True)
                    out = 0.5*(out1+out2)
                    out = torch.argmax(torch.softmax(
                        out, dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:ind + batch_size, ...] = pred
            else:
                slice = image[ind:, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net.eval()
                
                with torch.no_grad():
                    out1,out2  = net(input,mutli = True)
                    out = 0.5*(out1+out2)
                    out = torch.argmax(torch.softmax(
                        out, dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:, ...] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()

        with torch.no_grad():
            out1,out2  = net(input,mutli = True)
            out = 0.5*(out1+out2)
            out = torch.argmax(torch.softmax(
                out, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    case = case.replace(".h5", "")
    
    metric_list = []
    for i in range(1, FLAGS.num_classes):
        metric_list.append(calculate_metric_percase_save(
            prediction == i, label == i,spacing=(spacing[2], spacing[0], spacing[1])))
        
        
    #print(image.shape)
    #print(label.shape)
    #print(prediction.shape)
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing(spacing)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.int16))
    prd_itk.SetSpacing(spacing)
    lab_itk = sitk.GetImageFromArray(label.astype(np.int16))
    lab_itk.SetSpacing(spacing)
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))
    return np.array(metric_list)



def test_single_volume_generator(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    assert len(image.shape) == 3
    y_prediction = np.zeros_like(label)
    s_prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(
            slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            _,_,_,S, S_refine = net(input)

            S = torch.mean(S, dim=0)
            S_refine = torch.mean(S_refine, dim=0)

            y_out = torch.argmax(torch.softmax(S, dim=1), dim=1).squeeze(0)
            y_out = y_out.cpu().detach().numpy()
            pred = zoom(
                y_out, (x / patch_size[0], y / patch_size[1]), order=0)
            y_prediction[ind] = pred

            s_out = torch.argmax(S_refine, dim=1).squeeze(0)
            s_out = s_out.cpu().detach().numpy()
            pred = zoom(
                s_out, (x / patch_size[0], y / patch_size[1]), order=0)
            s_prediction[ind] = pred

    metric_list_y = []
    metric_list_s = []

    for i in range(1, classes):
        metric_list_y.append(calculate_metric_percase(
            y_prediction == i, label == i))
        metric_list_s.append(calculate_metric_percase(
            s_prediction == i, label == i))

    return metric_list_y, metric_list_s




def test_single_volume_ensemble(image, label, net1, net2, net3, net4, net5, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net1.eval()
            net2.eval()
            net3.eval()
            net4.eval()
            net5.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    ((net1(input) + net2(input) + net3(input) + net4(input) + net5(input)) / 5), dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net1.eval()
        net2.eval()
        net3.eval()
        net4.eval()
        net5.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                ((net1(input) + net2(input) + net3(input) + net4(input) + net5(input)) / 5), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output_main, _, _, _ = net(input)
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_cct(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output_main = net(input)[0]
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list



def test_single_volume_MIMO(image, label, net, classes, E, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction1 = np.zeros_like(label)
        prediction2 = np.zeros_like(label)
        prediction3 = np.zeros_like(label)

        prediction6 = np.zeros_like(label)

        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                B,C,H, W = list(input.shape)
                input = torch.stack([input] * E, dim=3)

                input = input.view(B,C,H,W*E)
                out = net(input)
                out = out.view(B, classes, H, W, E)
                out = torch.softmax(out, dim=1)

                #print(out.shape)

                out1 = out[0, ..., 0]
                out2 = out[0, ..., 1]
                out3 = out[0, ..., 2]

                out6 = torch.mean(out, dim=4).squeeze(0)


                out1 = torch.argmax(out1, dim=0)
                out1 = out1.cpu().detach().numpy()
                pred1 = zoom(
                    out1, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction1[ind] = pred1

                out2 = torch.argmax(out2, dim=0)
                out2 = out2.cpu().detach().numpy()
                pred2 = zoom(
                    out2, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction2[ind] = pred2

                out3 = torch.argmax(out3, dim=0)
                out3 = out3.cpu().detach().numpy()
                pred3 = zoom(
                    out3, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction3[ind] = pred3


                out6 = torch.argmax(out6, dim=0)
                out6 = out6.cpu().detach().numpy()
                pred6 = zoom(
                    out6, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction6[ind] = pred6


    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list1 = []
    for i in range(1, classes):
        metric_list1.append(calculate_metric_percase(
            prediction1 == i, label == i))

    metric_list2 = []
    for i in range(1, classes):
        metric_list2.append(calculate_metric_percase(
            prediction2 == i, label == i))


    metric_list3 = []
    for i in range(1, classes):
        metric_list3.append(calculate_metric_percase(
            prediction3 == i, label == i))


    metric_list6 = []
    for i in range(1, classes):
        metric_list6.append(calculate_metric_percase(
            prediction6 == i, label == i))


    return metric_list1, metric_list2, metric_list3, metric_list6


def test_single_volume_MIMO_c(image, label, net, classes, E, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction1 = np.zeros_like(label)
        prediction2 = np.zeros_like(label)
        prediction3 = np.zeros_like(label)

        prediction6 = np.zeros_like(label)

        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                B, C, H, W = list(input.shape)
                input = torch.stack([input] * E, dim=4)

                out = net(input)
                out = torch.softmax(out, dim=1) # (B, C, H, W, E)

                #print(out.shape)

                out1 = out[0, ..., 0]
                out2 = out[0, ..., 1]
                out3 = out[0, ..., 2]

                out6 = torch.mean(out, dim=4).squeeze(0)


                out1 = torch.argmax(out1, dim=0)
                out1 = out1.cpu().detach().numpy()
                pred1 = zoom(
                    out1, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction1[ind] = pred1

                out2 = torch.argmax(out2, dim=0)
                out2 = out2.cpu().detach().numpy()
                pred2 = zoom(
                    out2, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction2[ind] = pred2

                out3 = torch.argmax(out3, dim=0)
                out3 = out3.cpu().detach().numpy()
                pred3 = zoom(
                    out3, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction3[ind] = pred3


                out6 = torch.argmax(out6, dim=0)
                out6 = out6.cpu().detach().numpy()
                pred6 = zoom(
                    out6, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction6[ind] = pred6


    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list1 = []
    for i in range(1, classes):
        metric_list1.append(calculate_metric_percase(
            prediction1 == i, label == i))

    metric_list2 = []
    for i in range(1, classes):
        metric_list2.append(calculate_metric_percase(
            prediction2 == i, label == i))


    metric_list3 = []
    for i in range(1, classes):
        metric_list3.append(calculate_metric_percase(
            prediction3 == i, label == i))


    metric_list6 = []
    for i in range(1, classes):
        metric_list6.append(calculate_metric_percase(
            prediction6 == i, label == i))


    return metric_list1, metric_list2, metric_list3, metric_list6