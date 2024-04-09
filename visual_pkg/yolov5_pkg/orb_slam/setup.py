from setuptools import find_packages, setup

package_name = 'orb_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kuavo',
    maintainer_email='yangxiaokang@lejurobot.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "YOLO=orb_slam.YOLO:main",
            "color_map=orb_slam.color_map:main",
            "depth_color=orb_slam.depth_color:main"
        ],
    },
)
