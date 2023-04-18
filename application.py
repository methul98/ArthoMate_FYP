import base64
import io
import pickle
from PIL import Image
from flask import Flask, render_template, request,send_file,send_from_directory
from jinja2 import Environment
from index import *
import os
import math
import subprocess


application=Flask(__name__)
app=application 
app.secret_key = 'ArthoMate'

# Add the b64encode filter to the Jinja2 environment
env = Environment()
env.filters['b64encode'] = base64.b64encode
#home page
@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict')#Here I am uploading the image 
def predict():

    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/result',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        img=Image.open(request.files['uploadImage'])
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='png')
        img_byte_arr = img_byte_arr.getvalue()

        encoded_img = base64.b64encode(img_byte_arr).decode('utf-8')

        #From here onwards the coding for the rest of the works
        #1.Conversion from 2D to 3D shape
        image, depth_map = predict_depth(img)
        mesh_path = generate_mesh(image, depth_map,10)
        #1.1COnverting the obj file to png
        #2.Upload the image classification model
        #2.1 Image classification model
        res1=model1(img)

        #2.2Second model
        values=model2_val(img)
        with open('model_firstml','rb') as f:
             model2=pickle.load(f)
        res2=model2.predict(values)[0]


        # #2.3 Third Model
        with open('model_secondml','rb') as f:
            model3=pickle.load(f)
        res3 = model3.predict(values)[0]

        # #combine all the resutls
        fin_result= 0.15*res1+0.15*res2+0.70*res3
        print(fin_result)
        iff=(math.ceil(fin_result)-fin_result)
        if iff < 0.7:
            print("hello")
            fin_result=math.ceil(fin_result)
        else:
            fin_result = math.floor(fin_result)
        print(fin_result)
        return render_template('result.html',img_data=encoded_img,result=fin_result)
       
    return render_template('predict.html')

@app.route('/final_result',methods=['POST'])
def final_result():
    if request.method=='POST':
        doctrorRes=request.form['DoctorResult']
        modelRes=request.form['ModelResult']
        print(doctrorRes,modelRes)
        return render_template('final_result.html',model=modelRes,doctor=doctrorRes)
    
    return ''

@app.route('/download')
def download():
    filename = 'mesh.obj'
    directory = app.root_path + '/static/obj_files/'
    path = directory + filename
    return send_file(path, as_attachment=True)

@app.route('/preview', methods=['POST'])
def preview():
    filename = 'X_RAY.pcd'  # Replace with the actual filename of the .pcd file
    file_path = os.path.join(app.root_path, 'static/pcd_files', filename)
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])

    return ''



if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
