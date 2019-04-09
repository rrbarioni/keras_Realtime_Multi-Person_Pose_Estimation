from model.model_cmu_resnet50 import get_testing_model

weights_path = 'training/results/cmu_1stage_resnet50_res3d/model.h5'
model = get_testing_model()
model.save(weights_path)