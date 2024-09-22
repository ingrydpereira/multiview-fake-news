def train_autoencoder_7_views(model,
                              x_train_view1,
                              x_train_view2,
                              x_train_view3,
                              x_train_view4,
                              x_train_view5,
                              x_train_view6,
                              x_train_view7,
                              latent_dim = 60,
                              max_epochs = 100,
                              batch_size = 2000):
    print("Start train autoencoder")

    #Define parameters
    input_dim = [len(x_train_view1[1]),
                 len(x_train_view2[1]),
                 len(x_train_view3[1]),
                 len(x_train_view4[1]),
                 len(x_train_view5[1]),
                 len(x_train_view6[1]),
                 len(x_train_view7[1])]
    
    print(input_dim)

    #Define models
    ae = model(input_dim=input_dim, z_dim=latent_dim)

    ae.fit(x_train_view1,
           x_train_view2,
           x_train_view3,
           x_train_view4,
           x_train_view5,
           x_train_view6,
           x_train_view7,
           max_epochs=max_epochs,
           batch_size=batch_size)

    print("End train autoencoder")
    return ae

def train_autoencoder_6_views(model,
                              x_train_view1,
                              x_train_view2,
                              x_train_view3,
                              x_train_view4,
                              x_train_view5,
                              x_train_view6,
                              latent_dim = 60,
                              max_epochs = 100,
                              batch_size = 2000):
    print("Start train autoencoder")

    #Define parameters
    input_dim = [len(x_train_view1[1]),
                 len(x_train_view2[1]),
                 len(x_train_view3[1]),
                 len(x_train_view4[1]),
                 len(x_train_view5[1]),
                 len(x_train_view6[1])]

    #Define models
    ae = model(input_dim=input_dim, z_dim=latent_dim)

    ae.fit(x_train_view1,
           x_train_view2,
           x_train_view3,
           x_train_view4,
           x_train_view5,
           x_train_view6,
           max_epochs=max_epochs,
           batch_size=batch_size)

    print("End train autoencoder")
    return ae

def train_autoencoder_5_views(model,
                              x_train_view1,
                              x_train_view2,
                              x_train_view3,
                              x_train_view4,
                              x_train_view5,
                              latent_dim = 60,
                              max_epochs = 100,
                              batch_size = 2000):
    print("Start train autoencoder")

    #Define parameters
    input_dim = [len(x_train_view1[1]),
                 len(x_train_view2[1]),
                 len(x_train_view3[1]),
                 len(x_train_view4[1]),
                 len(x_train_view5[1])]
    
    print(input_dim)

    #Define models
    ae = model(input_dim=input_dim, z_dim=latent_dim)

    ae.fit(x_train_view1,
           x_train_view2,
           x_train_view3,
           x_train_view4,
           x_train_view5,
           max_epochs=max_epochs,
           batch_size=batch_size)

    print("End train autoencoder")
    return ae

def train_autoencoder_4_views(model,
                              x_train_view1,
                              x_train_view2,
                              x_train_view3,
                              x_train_view4,
                              latent_dim = 60,
                              max_epochs = 100,
                              batch_size = 2000):
    print("Start train autoencoder")

    #Define parameters
    input_dim = [len(x_train_view1[1]),
                 len(x_train_view2[1]),
                 len(x_train_view3[1]),
                 len(x_train_view4[1])]
    
    print(input_dim)

    #Define models
    ae = model(input_dim=input_dim, z_dim=latent_dim)

    ae.fit(x_train_view1,
           x_train_view2,
           x_train_view3,
           x_train_view4,
           max_epochs=max_epochs,
           batch_size=batch_size)

    print("End train autoencoder")
    return ae

def train_autoencoder_3_views(model,
                              x_train_view1,
                              x_train_view2,
                              x_train_view3,
                              latent_dim = 60,
                              max_epochs = 100,
                              batch_size = 2000):
    print("Start train autoencoder")

    #Define parameters
    input_dim = [len(x_train_view1[1]),
                 len(x_train_view2[1]),
                 len(x_train_view3[1])]
    
    print(input_dim)

    #Define models
    ae = model(input_dim=input_dim, z_dim=latent_dim)

    ae.fit(x_train_view1,
           x_train_view2,
           x_train_view3,
           max_epochs=max_epochs,
           batch_size=batch_size)

    print("End train autoencoder")
    return ae

def train_autoencoder_2_views(model,
                              x_train_view1,
                              x_train_view2,
                              latent_dim = 60,
                              max_epochs = 100,
                              batch_size = 2000):
    print("Start train autoencoder")

    #Define parameters
    input_dim = [len(x_train_view1[1]),
                 len(x_train_view2[1])]
    
    print(input_dim)

    #Define models
    ae = model(input_dim=input_dim, z_dim=latent_dim)

    ae.fit(x_train_view1,
           x_train_view2,
           max_epochs=max_epochs,
           batch_size=batch_size)

    print("End train autoencoder")
    return ae
