from django.shortcuts import render, redirect
# from pyresparser import ResumeParser
from .models import Resume, UploadResumeModelForm
from django.contrib import messages
from django.conf import settings
from django.db import IntegrityError
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, FileResponse, Http404
import os
import pdf2docx
from docx2pdf import convert
from resume_parser.resume_parser import ResumeParser
from PIL import Image
def index(request):
    return render(request,'index.html',{'nbar':'home'})

def tool(request):
    return render(request,'tools.html',{'nbar':'tool'})
def contact(request):
    return render(request,'contact.html',{'nbar':'contact'})
def upload(request):
    upload_name = request.GET.get('upload_name')

    if request.method == 'POST':
        file_name = request.POST['upload']

        file = request.FILES['image']
        filename = file.name
        extension = filename.split('.')[1]
        if(file_name=='img2pdf' and (extension == 'jpg' or extension == 'jpeg' or extension=='png')):
            temp_path = f'/tmp/{filename}'
            with open(temp_path,'wb') as f:
                f.write(file.read())
            pdf_path = f'/tmp/{filename.split(".")[0]}.pdf'
            # print(pdf_path)
            img = Image.open(temp_path)
            im = img.convert('RGB')
            im.save(pdf_path)

            fs = FileSystemStorage(pdf_path)
            response = FileResponse(fs.open(pdf_path,'rb'),content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{filename.split(".")[0]}.pdf"'

            return response
        elif (file_name=='pdf2doc' and (extension == "pdf")):
            temp_path = f'/tmp/{filename}'
            with open(temp_path,'wb') as f:
                f.write(file.read())
            docx_path = f'/tmp/{filename.split(".")[0]}.docx'
            pdf_to_docx = pdf2docx.parse(temp_path,docx_path,start=0,end=None)
            fs = FileSystemStorage(docx_path)
            response = FileResponse(fs.open(docx_path,'rb'),content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
            response['Content-Disposition'] = f'attachment; filename="{filename.split(".")[0]}.docx"'

            return response
        elif (file_name=='doc2pdf' and (extension=='doc' or extension=='docx')):
            temp_path = f'/tmp/{filename}'
            with open(temp_path,'wb') as f:
                f.write(file.read())
            pdf_path = f'/tmp/{filename.split(".")[0]}.pdf'
            convert(temp_path,pdf_path)

            fs = FileSystemStorage(pdf_path)
            response = FileResponse(fs.open(pdf_path,'rb'),content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{filename.split(".")[0]}.pdf"'
            
            return response
    return render(request,'upload.html',{'upload':upload_name})

def homepage(request):
    if request.method == 'POST':
        Resume.objects.all().delete()
        file_form = UploadResumeModelForm(request.POST, request.FILES)
        files = request.FILES.getlist('resume')
        resumes_data = []
        if file_form.is_valid():
            for file in files:
                try:
                    # saving the file
                    resume = Resume(resume=file)
                    resume.save()
                    
                    # extracting resume entities
                    parser = ResumeParser(os.path.join(settings.MEDIA_ROOT, resume.resume.name))
                    data = parser.get_extracted_data()
                    # print(data)
                    resumes_data.append(data)
                    resume.name               = data.get('name')
                    resume.email              = data.get('email')
                    resume.mobile_number      = data.get('mobile_number')
                    if data.get('education') is not None:
                        resume.education      = ', '.join(data.get('education'))
                    else:
                        # print('here')
                        resume.education      = None
                    resume.company_name        = data.get('company_names')
                    resume.college_name       = data.get('college_name')
                    resume.designation        = data.get('designation')
                    resume.total_experience   = data.get('total_experience')
                    if data.get('skills') is not None:
                        resume.skills         = ', '.join(data.get('skills'))
                    else:
                        resume.skills         = None
                    if data.get('experience') is not None:
                        resume.experience     = ', '.join(data.get('experience'))
                    else:
                        resume.experience     = None
                    resume.save()
                except IntegrityError:
                    messages.warning(request, 'Duplicate resume found:', file.name)
                    return redirect('homepage')
            resumes = Resume.objects.all()
            messages.success(request, 'Resumes uploaded!')
            context = {
                'resumes': resumes,
            }
            return render(request, 'base.html', context)
    else:
        form = UploadResumeModelForm()
    return render(request, 'base.html', {'form': form,'nbar':'base'})