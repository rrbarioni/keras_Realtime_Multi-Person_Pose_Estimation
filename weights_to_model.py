from model.model_cmu_resnet50 import get_testing_model

weights_path = 'training/results/cmu_1stage_resnet50_res3d/weights.h5'
model_path = 'training/results/cmu_1stage_resnet50_res3d/model.h5'
model = get_testing_model()
model.load_weights(weights_path)
model.save(model_path)