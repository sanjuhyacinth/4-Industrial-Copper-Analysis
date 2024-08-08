# libraries:
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle

#set up page configuration for streamlit
icon='https://static.vecteezy.com/system/resources/previews/029/568/133/original/copper-icon-vector.jpg'
st.set_page_config(page_title='Industrial copper',page_icon=icon,initial_sidebar_state='expanded',
                        layout='wide',menu_items={"about":'This streamlit application was developed by Sanju Hyacinth C'})

title_text = '''<h1 style='font-size: 45px;color:#ec7345;text-align: center;'>Industrial Copper Modelling</h1>'''
st.markdown(title_text, unsafe_allow_html=True)

#set up the sidebar with optionmenu
with st.sidebar:
    selected = option_menu(None,
                            options=["Home","About","Predictive Analytics","Inferences"],
                            icons=["house-fill","info-circle-fill","file-bar-graph-fill","lightbulb-fill"],
                            default_index=0,
                            orientation="vertical",)

# set up the information for 'Home' menu
if selected == 'Home':
    st.write('')
    title_text = '''<h3 style='font-size: 35px;color:#ec7345;text-align: center;'>What is Copper?</h3>'''
    st.markdown(title_text, unsafe_allow_html=True)
    st.write('')
    col1,col2=st.columns([2,1.5], gap = "large")
    with col1:
        st.markdown('''<h5 style='color:#44454A;font-size:20px'> Copper is a reddish brown metal that is found in abundance all around the world, 
                    while the top three producers are Chile, Peru, and China. Historically, copper was the first metal to be worked 
                    by human hands. When we discovered that it could be hardened with tin to make bronze around 3000 BC,
                    the Bronze Age was ushered in, changing the course of humanity.''',unsafe_allow_html=True)
        st.write('')
        st.link_button(':violet[Copper - Wiki]',url='https://en.wikipedia.org/wiki/Copper')
        st.write('')
        st.write('')
    with col2:
        st.image('https://www.mining.com/wp-content/uploads/2023/12/AdobeStock_648674620-1024x683.jpeg',caption="Copper wires - google image",width = 330)

    title_text = '''<h3 style='font-size: 35px;color:#ec7345;text-align: center;'>India's graph in Copper</h3>'''
    st.markdown(title_text, unsafe_allow_html=True)
    st.write('')
    left, right = st.columns([2,2], gap = "medium")
    with right:
        st.write('')
        st.markdown('''<h5 style='font-size:20px;color:#44454A'> From a net exporter of 335,000 tonnes in 2017-18, 
                    India became a net importer of copper, first in 2018-19 and the trend remains unaltered till 2021-22. 
                    During the April-October period of the current fiscal also, Indian import at 88,000 tonnes was higher 
                    than exports of 16,000 tonnes. The image displays the numbers clearly.''',unsafe_allow_html=True)
        st.write('')
        st.write('')
    with left:
        st.image('https://static.theprint.in/wp-content/uploads/2023/01/ANI-20230111130414.jpg',width=425)
    st.write('')
    title_text = '''<h3 style='font-size: 35px;color:#ec7345;text-align: center;'>What is Copper used for?</h3>'''
    st.markdown(title_text, unsafe_allow_html=True)
    st.write('')
    left, right = st.columns([1.5,1], gap = "large")
    with right:
        st.image('https://sterlitecopper.com/blog/wp-content/uploads/2018/07/01.png',width=300)
    with left:
        st.markdown('''<h5 style='font-size:20px;color:#44454A'> Copper is used in almost all the industries. The copper
                    consumption percentage in various industries in India is given in the image.<br><br>
                    Presently, copper is used in a bunch of fields like <br>''',unsafe_allow_html=True)
        st.markdown('''<h5 style='font-size:20px;color:#13a886'>Building construction, <br>
                    Power generation and transmission, <br>
                    Electronic product manufacturing, <br>
                    Production of industrial machinery and <br>
                    Transportation vehicles.</h5>''', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.markdown('''<h4 style='font-size: 30px;color:#ec7345;text-align: left;'>More about Copper uses</h4>''',unsafe_allow_html=True)
    st.markdown('''<h5 style='font-size:18px;color:grey'> Click on the expanders to learn more.''',unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns([1,1,1,1], gap="small")
    with c1:
        with st.container():
            with st.expander(':violet[**Electrical Wiring**]'):
                st.markdown('''Copper is used in virtually all electrical wiring (except for power lines, 
                            which are made with aluminum) because it is the second most electrically conductive metal aside from silver 
                            which is much more expensive. In addition to being widely available and inexpensive, it is malleable and easy to
                            stretch out into very thin, flexible but strong wires, making it ideal to use in electrical infrastructure.
                            ''',unsafe_allow_html=True)
    with c2:
        with st.container():
            with st.expander(':violet[**Construction, Piping, & Design**]'):
                st.markdown('''Copper has been used as construction material for centuries. 
                            Copper is still used today in architecture due to its corrosion resistance, easy workability, 
                            and attractiveness; copper sheets make a beautiful roofing material and other exterior features on buildings.
                            On the interior, copper is used in door handles, trim, vents, railings, kitchen appliances and cookware, 
                            lighting fixtures, and more.''',unsafe_allow_html=True)
    
    with c3:
        with st.container():
            with st.expander(':violet[**Transportation**]'):
                st.markdown('''Aside from the copper wiring used in the electrical components of modern cars, copper 
                            and brass have been the industry standard for oil coolers and radiators since the 1970s. Alloys that include copper are used 
                            in the locomotive and aerospace industries as well. As demand for electric cars and other forms of transportation increases,
                            demand for copper components also increases.''',unsafe_allow_html=True)

    with c4: 
        with st.container():
            with st.expander(':violet[**Other Uses**]'):
                st.markdown('''Because copper is a beautiful, easily worked material, it is used in art such as copper
                            sheet metal sculptures, jewelry, signage, musical instruments, cookware, and more. The Statue of Liberty is plated with more than
                            80 tons of copper, which gives her the characteristic pale green patina. Due to its antimicrobial properties, copper is also starting 
                            to gain popularity for high-touch items such as faucets, doorknobs, latches, railings, counters, hooks, handles, and other public 
                            surfaces that tend to gather a lot of germs.''',unsafe_allow_html=True)

    st.write('')  
    title_text = '''<h3 style='font-size: 30px;color:#ec7345;text-align: left;'>Video references on Copper</h3>'''
    st.markdown(title_text, unsafe_allow_html=True)
    st.write('')  
    col1,col2,col3=st.columns(3)

    with col1:
        st.video('https://www.youtube.com/watch?v=gqmkiPPIsUQ&pp=ygUNIGFib3V0IGNvcHBlcg%3D%3D')
    with col2:
        st.video('https://www.youtube.com/watch?v=AgRYHT6WFV0&pp=ygUTIGNvcHBlciBpbiBpbmR1c3RyeQ%3D%3D')
    with col3:
        st.video('https://youtu.be/g8Nar1s5UgM?si=ALCKDdRJPgQunIhi')

#set up information for 'About' menu 
if selected == "About":
    c1,c2 = st.columns([1,2], gap="medium")
    with c1:
        st.write('')
        text = '''<h4 style='font-size: 28px;color:#13a886;text-align: left;'>Project Title :</h4>'''
        st.markdown(text, unsafe_allow_html=True)
        st.markdown('''<h4 style='font-size: 28px;color:#13a886;text-align: left;'>Project Domain :</h4>''',unsafe_allow_html=True)
        st.markdown('''<h4 style='font-size: 28px;color:#13a886;text-align: left;'>Skills & Technologies :</h4>''',unsafe_allow_html=True)
    with c2:
        st.write('')
        st.write('')
        st.markdown('''<h5 style= 'font-size: 22px;color:#44454A;'> Industrial Copper Modeling''',unsafe_allow_html=True)
        st.write('')
        st.markdown('''<h5 style= 'font-size: 22px;color:#44454A;'> Manufacturing Industry''',unsafe_allow_html=True)
        st.write('')
        st.markdown('''<h5 style= 'font-size: 22px;color:#44454A;'> Python scripting, Data Preprocessing, Machine learning, EDA, Streamlit ''',unsafe_allow_html=True)
    st.write('')
    st.markdown('''<h4 style='font-size: 28px;color:#13a886;text-align: left;'>Overview </h4>''',unsafe_allow_html=True)
    st.write('')
    c1,c2 = st.columns([2,2], gap = "medium")
    with c1:
        st.markdown('''<h5 style= 'font-size: 22px;color:#ec7345;'>1️⃣ Data Preprocessing: ''',unsafe_allow_html=True)
        st.markdown('''    
                    <li>Loaded the copper CSV into a DataFrame. <br>              
                    <li>Cleaned and filled missing values, addressed outliers, and adjusted data types.  <br>           
                    <li>Analyzed data distribution and treated skewness.<br>''',unsafe_allow_html=True)
        st.write('')
    with c2:
        st.markdown('''<h5 style= 'font-size: 22px;color:#ec7345;'>2️⃣ Feature Engineering: ''',unsafe_allow_html=True)
        st.markdown('''<li>Assessed feature correlation to identify potential multicollinearity <br>
                    <li>Encoded categorical variables to use in prediction''',unsafe_allow_html=True)
    
    c1,c2 = st.columns([2,2], gap="medium")
    with c1:
        st.markdown('''<h5 style= 'font-size: 22px;color:#ec7345;'>3️⃣ Modeling: ''',unsafe_allow_html=True)
        st.markdown('''
                    <li >Built a regression model for selling price prediction.
                    <li>Built a classification model for status prediction.
                    <li>Pickled the trained models for deployment.''',unsafe_allow_html=True)
    with c2:
        st.markdown('''<h5 style= 'font-size: 22px;color:#ec7345;'>4️⃣ Streamlit Application: ''',unsafe_allow_html=True)
        st.markdown('''
                    <li>Developed a user interface for interacting with the models.
                    <li>Predicted selling price and status based on user input. <br>''',unsafe_allow_html=True)
    st.write('')   
    st.markdown('''<h4 style='font-size: 28px;color:#13a886;text-align: left;'>Connect with me :</h4>''',unsafe_allow_html=True)
    st.markdown('##### Linkedin: www.linkedin.com/in/sanju-hyacinth/')
    st.markdown('##### GitHub : https://github.com/sanjuhyacinth') 

#user input values for selectbox and encoded for respective features
class option():
    
    # our country values are encoded ordered from the most value counts to the least
    country_values_enc=[10., 1., 0., 2., 5., 3., 13., 9., 4., 7., 11., 6., 8., 12., 16., 14., 15.]

    status_values=['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM','Wonderful', 'Revised',
            'Offered', 'Offerable']

    status_encoded = {'Lost':0, 'Won':1, 'Draft':2, 'To be approved':3, 'Not lost for AM':4,'Wonderful':5, 'Revised':6,
                    'Offered':7, 'Offerable':8}
    
    item_type_values=['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']

    item_type_encoded = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}

    application_values=[2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 40.0,
                41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
    
    product_ref_values=[611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407,
                164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642,
                1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026,
                1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

#set up information for the 'get prediction' menu
if selected == 'Predictive Analytics':
    title_text = '''<h2 style='font-size: 28px;text-align: center;color:#13a886;'>Copper Selling Price and Status Prediction</h2>'''
    st.markdown(title_text, unsafe_allow_html=True)
    st.write('')
    
    #set up option menu for selling price and status menu
    select=option_menu('',options=["SELLING PRICE","STATUS"],
                                    icons=["cash", "toggles"],
                                    orientation='horizontal',)


    if select == 'SELLING PRICE':
        st.markdown("<h5 style=color:grey>To predict the selling price of copper, please provide the following information:",unsafe_allow_html=True)
        st.write('')

        # creted form to get the user input 
        with st.form('Price Prediction'):
            col1,col2=st.columns(2)
            with col1:
                # st.date gives the ability to choose a date from the calendar view
                item_date = st.date_input(label='Item Date',format='DD-MM-YYYY')
                # st.selectbox allows us to choose from the options we have given in the list above
                country=st.selectbox(label='Country',options = option.country_values_enc,index=None)
                item_type=st.selectbox(label='Item Type',options = option.item_type_values,index=None)
                # st.number_input allows to input numbers or add value with + sign from min value given.
                customer = st.number_input('Customer ID',min_value=10000)
                thickness=st.number_input(label='Thickness',min_value=0.1)
                quantity=st.number_input(label='Quantity',min_value=0.1)
                
            with col2:
                # st.date gives the ability to choose a date from the calendar view
                delivery_date=st.date_input(label='Delivery Date',format='DD-MM-YYYY')
                # st.selectbox allows us to choose from the options we have given in the list above
                status=st.selectbox(label='Status',options=option.status_values,index=None)
                product_ref=st.selectbox(label='Product Ref',options = option.product_ref_values,index=None)
                application=st.selectbox(label='Application',options = option.application_values,index=None)
                # st.number_input allows to input numbers or add value with + sign from min value given.
                width=st.number_input(label='Width',min_value=1.0)
                st.markdown('<br>', unsafe_allow_html=True)
                # st.form_submit_button is an exclusive function for st.form
                button= st.form_submit_button('PREDICT PRICE', use_container_width=True)

        if button:
            #check whether user fill all required fields
            if not all([country, item_type, application, product_ref,
                        customer, status, quantity, width, thickness]):
                st.error("Please fill in all required fields.")

            else:
                
                #opened pickle model and predict the selling price with user data
                with open('RandomForestRegressor.pkl','rb') as files:
                    predict_model=pickle.load(files)

                # customize the user data to fit the feature 
                status=option.status_encoded[status]
                item_type=option.item_type_encoded[item_type]

                delivery_time_taken=abs((item_date - delivery_date).days)

                quantity_log=np.log(quantity)
                thickness_log=np.log(thickness)

                #predict the selling price with regressor model
                user_data=np.array([[customer, country, status, item_type ,application, width, product_ref,
                                    delivery_time_taken, quantity_log, thickness_log ]])
                
                pred=predict_model.predict(user_data)

                selling_price=np.exp(pred[0])

                #display the predicted selling price 
                st.subheader(f":green[Predicted Selling Price :] {selling_price:.2f}") 
                

    if select == 'STATUS':
        st.markdown("<h5 style=color:grey;>To predict the status of copper, please provide the following information:",unsafe_allow_html=True)
        st.write('')

        #creted form to get the user input 
        with st.form('Status Classifier'):
            col1,col2=st.columns(2)

            with col1:
                item_date = st.date_input(label='Item Date',format='DD-MM-YYYY')
                country=st.selectbox(label='Country',options=option.country_values_enc,index=None)
                item_type=st.selectbox(label='Item Type',options=option.item_type_values,index=None)
                thickness=st.number_input(label='Thickness',min_value=0.1)
                application=st.selectbox(label='Application',options=option.application_values,index=None)
                product_ref=st.selectbox(label='Product Ref',options=option.product_ref_values,index=None)

            with col2:
                delivery_date=st.date_input(label='Delivery Date',format='DD-MM-YYYY')
                customer=st.number_input('Customer ID',min_value=10000)
                quantity=st.number_input(label='Quantity',min_value=0.1)
                width=st.number_input(label='Width',min_value=1.0)
                selling_price=st.number_input(label='Selling Price',min_value=0.1)
                st.markdown('<br>', unsafe_allow_html=True)
                # form submission button to show predicted status
                button=st.form_submit_button('PREDICT STATUS',use_container_width=True)

        if button:
            #check whether user fill all required fields
            if not all([item_date, delivery_date, country, item_type, application, product_ref,
                        customer,quantity, width, thickness,selling_price]):
                st.error("Please fill in all required fields.")

            else:
                #opened pickle model and predict status with user data
                with open('ExtraTreesClassifier.pkl','rb') as files:
                    model=pickle.load(files)

                # customize the user data to fit the feature 
                item_type=option.item_type_encoded[item_type]

                delivery_time_taken=abs((item_date - delivery_date).days)

                quantity_log=np.log(quantity)
                thickness_log=np.log(thickness)
                selling_price_log=np.log(selling_price)

                #predict the status with classifier model
                user_data=np.array([[customer, country, item_type ,application, width, product_ref,
                                    delivery_time_taken, quantity_log, thickness_log, selling_price_log ]])
                
                status=model.predict(user_data)

                #display the predicted status 
                if status==1:
                    st.subheader(f":green[Status of the copper : ] Won")

                else:
                    st.subheader(f":red[Status of the copper :] Lost")

if selected == 'Inferences':
    title_text = '''<h2 style='font-size: 35px;text-align: center;color:#13a886;'>Inferences</h2>'''
    st.markdown(title_text, unsafe_allow_html=True)
    st.markdown('''In the previous section of Predictive Analytics, we saw the price and status predictions based on
                user inputs. Even though we were able to get outcomes for both results, the outcome, especially for the status,
                seems to have some concerns.<br>''',unsafe_allow_html=True)
    st.markdown('''<h4 style='color:#13a886;'>
                Selling Price Prediction:''',unsafe_allow_html=True)
    st.markdown('''Although we have a big dataset with **181673 entries**, the composition of data is not very consistent throughout.
                The data is :red[noisy] for some variables, meaning all over the place, which would have to be cleaned before using 
                prediction alorithms. Our selling price field for example, had high variance with negative values in it. Since there is skewness
                of this type in the data, we have chosen to work with only :red[tree based regressors] which can handle the noise 
                in the dataset through splitting other than :red[Linear regression] which would generally need data that is consistent.<br>
                Our tree based regressor models, in basic conditions, worked well on this data especially our :red[Random Forest Regressor] model
                which showed about **92%** accuracy.
                <br>
                ''',unsafe_allow_html=True)
    st.markdown('''<h4 style='color:#13a886;'>
                Status Prediction:''',unsafe_allow_html=True)
    st.markdown('''Our next prediction to be made was the :red[Status] field. Here we have used techniques to make the data better. We have made
                use of only 2 statuses of Lost and Won for our prediction. We have done **oversampling** with SMOTE to balance out the status classes.
                Here too we used an ensemble of Random Forest, Extra Trees and XGB Classifier models, out of  which the :red[Extra Trees Classifier] model
                beat the Random forest very closely. The model's accuracy seems very high of roughly **98%** which seems almomst unnatural for a non-tuned
                model. <br> However, the challenge with the prediction is that most of the predictions made are of the :green[Won] status. Even though
                we have a list of variables that greatly affect the predictions, tuning them also seems less likely for us to arrive at a prediction of
                :red[Lost]. This is because our model is highly trained with most of the status as Won, even with the different supporting fields. In our example,
                we saw that the Item Type of SLAWR gave a lost status, since there is not much training made on that type with lost. Hence for tree based
                classifier models, it is import to get training made on large datatsets. 
                <br>
                ''',unsafe_allow_html=True)