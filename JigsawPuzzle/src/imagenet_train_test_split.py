import os
import shutil

def train_test_split(input_dir, output_dir, train_split=0.9):

    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    for i, folder in enumerate(os.listdir(input_dir)):
        images = os.listdir(os.path.join(input_dir, folder))

        if not os.path.exists(os.path.join(train_dir, folder)):
            os.mkdir(os.path.join(train_dir, folder))
            train_images = images[:round(len(images) * train_split)]
            for img in train_images:
                shutil.move(os.path.join(input_dir, folder, img),
                        os.path.join(train_dir, folder, img))

        if not os.path.exists(os.path.join(test_dir, folder)):
            os.mkdir(os.path.join(test_dir, folder))
            test_images = images[round(len(images) * train_split):]
            for img in test_images:
                shutil.move(os.path.join(input_dir, folder, img),
                        os.path.join(test_dir, folder, img))

        print(i)


def main():
    train_test_split('imagenet/all', 'imagenet')


if __name__ == '__main__':
    main()
