# Search Similar 3D Meshes

In this tutorial, we will learn how to build a 3D mesh search pipeline with Jina. In particular, we will be building a search pipeline for 3D models in GLB format.

Just like other data types, the 3D meshes search pipeline consists of **loading**, **encoding** and **indexing** the data. After data is indexed, we can then search them.


## Load GLB data

First, given a `glb` file, how do we load and craft the `glb` into a Document so that we can process and encode?  Let's use `trimesh` to build an executor for this.

```python
class GlbCrafter(Executor):
    def sample(self, mesh: trimesh.Scene):
        geo = list(mesh.geometry.values())
        return geo[0].sample(2048)
       
    @requests(on=['/index', '/search'])
    def craft(self, docs: DocumentArray, **kwargs):
        for d in docs:
            mesh = trimesh.load_mesh(d.uri)
            d.blob = self.sample(mesh)
```

We first load the data of each `glb` file into a Python object. We will use the `trimesh` package for this, which loads the `glb` data and represent them in the form of triangular meshes. The loaded object is of type `trimesh.Scene` which may contain one or more triangular mesh geometries. For simplicity, we select the first mesh gemoetry in the Scene and sample surfaces from the selected mesh geeomtry. The sampled surface will be made from 2048 points in 3D space and hence the shape of the `ndarray` representing each 3D model will be `(2048, 3)`.

## Encode 3D Model

Once we convert each `glb` model into a `ndarray`, encoding the inputs becomes straight forward. We will use our pre-trained `pointnet` to encode the data. The model looks like:

```python
def get_model(ckpt_path):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.python.framework.errors_impl import NotFoundError
    
    def conv_bn(x, filters):
        x = layers.Conv1D(filters, kernel_size=1, padding='valid')(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation('relu')(x)
    
    
    def dense_bn(x, filters):
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation('relu')(x)
    
    
    def tnet(inputs, num_features):
        class OrthogonalRegularizer(keras.regularizers.Regularizer):
            def __init__(self, num_features_, l2reg=0.001):
                self.num_features = num_features_
                self.l2reg = l2reg
                self.eye = tf.eye(self.num_features)
    
            def __call__(self, x):
                x = tf.reshape(x, (-1, self.num_features, self.num_features))
                xxt = tf.tensordot(x, x, axes=(2, 2))
                xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
                return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
            def get_config(self):
                return {'num_features': self.num_features,
                        'l2reg': self.l2reg,
                        'eye': self.eye.numpy()}
    
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)
    
        x = conv_bn(inputs, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = dense_bn(x, 128)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer='zeros',
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        return layers.Dot(axes=(2, 1))([inputs, feat_T])

    inputs = keras.Input(shape=(2048, 3))
    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnet')
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.get_layer(f'dense_1').output)
    intermediate_layer_model.load_weights(ckpt_path)
    return intermediate_layer_model
```

With the above model, we can then build our `pointnet` executor:

```python
class PNEncoder(Executor):
    def __init__(self, ckpt_path: str, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = get_model(ckpt_path=ckpt_path)

    @requests(on=['/index', '/search'])
    def encode(self, docs: DocumentArray, **kwargs):
        print('what')
        docs.embeddings = self.embedding_model.predict(docs.blobs)
```


## Index the data

Finally, let's also build a `MyIndexer` to index the data.

```python
class MyIndexer(Executor):
    _docs = DocumentArray()

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        self._docs.extend(docs)

    @requests(on='/search')
    def search(self, docs: DocumentArray, **kwargs):
        docs.match(self._docs, limit=5)
```

## Visualize 3D Model

Let's also build a Visualizer to visualize the results.

```python
class GlbVisualizer:
    def __init__(self, search_doc, matches=None, plot_matches=True):
        self.search_doc = search_doc
        self.matches = matches
        self.fig = plt.figure()
        self.plot_matches = plot_matches

    def visualize(self):
        subplot = 221 if self.plot_matches else 111
        self.visualize_3d_object(self.search_doc.uri, subplot, 'Search')
        if self.plot_matches:
            self.visualize_3d_object(self.matches[0].uri, 222, '1st Match')
            self.visualize_3d_object(self.matches[1].uri, 223, '2nd Match')
            self.visualize_3d_object(self.matches[2].uri, 224, '3rd Match')
        plt.show()

    def visualize_3d_object(self, uri, ax, title):
        import tempfile
        doc = Document(uri=uri)
        doc.convert_uri_to_buffer()
        with tempfile.NamedTemporaryFile(
                suffix='.glb', delete=True
        ) as glb_file:
            glb_file.write(doc.content)
            glb_file.flush()
            mesh = trimesh.load(glb_file.name)
            geo = list(mesh.geometry.values())
            mesh = geo[0]
            ax = self.fig.add_subplot(ax, projection='3d')
            ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1],
                            triangles=mesh.faces, Z=mesh.vertices[:, 2])
            ax.set_title(title)
```


## Index, Search and Visualize Data

Download the pre-trained PNEncoder model [here](https://github.com/jina-ai/example-3D-model/tree/main/executors/pn_encoder/ckpt) into `model/ckpt`. Also, store the index/search data in `data/`. We can then put the executors in flow and use the flow to perform indexing and searching. Finally, we use the `GlbVisualizer` built earlier to visualize our data.

```python

with Flow().add(uses=GlbCrafter).add(uses=PNEncoder, uses_with={'ckpt_path': 'model/ckpt/ckpt_True'}).add(uses=MyIndexer) as f:
    f.index(from_files('data/*.glb'))
    results = f.search(Document(uri='data/airplane_aeroplane_plane_13.glb'), return_results=True)
    doc = results[0].docs[0]
    visualizer = GlbVisualizer(doc, matches=doc.matches, plot_matches=True).visualize()

```


## Putting it all together

Combining the steps listed above and import the necessary dependencies, the following is the complete code.

```
import glob
import trimesh
from jina import Flow, Executor, DocumentArray, Document, requests
from jina.types.document.generators import from_files
import matplotlib.pyplot as plt

class GlbCrafter(Executor):
    def sample(self, mesh: trimesh.Scene):
        geo = list(mesh.geometry.values())
        return geo[0].sample(2048)
       
    @requests(on=['/index', '/search'])
    def craft(self, docs: DocumentArray, **kwargs):
        for d in docs:
            mesh = trimesh.load_mesh(d.uri)
            d.blob = self.sample(mesh)


def get_model(ckpt_path):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.python.framework.errors_impl import NotFoundError
    
    def conv_bn(x, filters):
        x = layers.Conv1D(filters, kernel_size=1, padding='valid')(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation('relu')(x)
    
    
    def dense_bn(x, filters):
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation('relu')(x)
    
    
    def tnet(inputs, num_features):
        class OrthogonalRegularizer(keras.regularizers.Regularizer):
            def __init__(self, num_features_, l2reg=0.001):
                self.num_features = num_features_
                self.l2reg = l2reg
                self.eye = tf.eye(self.num_features)
    
            def __call__(self, x):
                x = tf.reshape(x, (-1, self.num_features, self.num_features))
                xxt = tf.tensordot(x, x, axes=(2, 2))
                xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
                return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
            def get_config(self):
                return {'num_features': self.num_features,
                        'l2reg': self.l2reg,
                        'eye': self.eye.numpy()}
    
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)
    
        x = conv_bn(inputs, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = dense_bn(x, 128)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer='zeros',
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        return layers.Dot(axes=(2, 1))([inputs, feat_T])

    inputs = keras.Input(shape=(2048, 3))
    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnet')
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.get_layer(f'dense_1').output)
    intermediate_layer_model.load_weights(ckpt_path)
    return intermediate_layer_model

class PNEncoder(Executor):
    def __init__(self, ckpt_path: str, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = get_model(ckpt_path=ckpt_path)

    @requests(on=['/index', '/search'])
    def encode(self, docs: DocumentArray, **kwargs):
        print('what')
        docs.embeddings = self.embedding_model.predict(docs.blobs)


class MyIndexer(Executor):
    _docs = DocumentArray()

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        self._docs.extend(docs)

    @requests(on='/search')
    def search(self, docs: DocumentArray, **kwargs):
        docs.match(self._docs, limit=5)


class GlbVisualizer:
    def __init__(self, search_doc, matches=None, plot_matches=True):
        self.search_doc = search_doc
        self.matches = matches
        self.fig = plt.figure()
        self.plot_matches = plot_matches

    def visualize(self):
        subplot = 221 if self.plot_matches else 111
        self.visualize_3d_object(self.search_doc.uri, subplot, 'Search')
        if self.plot_matches:
            self.visualize_3d_object(self.matches[0].uri, 222, '1st Match')
            self.visualize_3d_object(self.matches[1].uri, 223, '2nd Match')
            self.visualize_3d_object(self.matches[2].uri, 224, '3rd Match')
        plt.show()

    def visualize_3d_object(self, uri, ax, title):
        import tempfile
        doc = Document(uri=uri)
        doc.convert_uri_to_buffer()
        with tempfile.NamedTemporaryFile(
                suffix='.glb', delete=True
        ) as glb_file:
            glb_file.write(doc.content)
            glb_file.flush()
            mesh = trimesh.load(glb_file.name)
            geo = list(mesh.geometry.values())
            mesh = geo[0]
            ax = self.fig.add_subplot(ax, projection='3d')
            ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1],
                            triangles=mesh.faces, Z=mesh.vertices[:, 2])
            ax.set_title(title)

with Flow().add(uses=GlbCrafter).add(uses=PNEncoder, uses_with={'ckpt_path': 'model/ckpt/ckpt_True'}).add(uses=MyIndexer) as f:
    f.index(from_files('data/*.glb'))
    results = f.search(Document(uri='data/airplane_aeroplane_plane_13.glb'), return_results=True)
    doc = results[0].docs[0]
    visualizer = GlbVisualizer(doc, matches=doc.matches, plot_matches=True).visualize()
```
